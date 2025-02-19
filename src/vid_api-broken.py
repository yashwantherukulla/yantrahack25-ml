#Not Working as of now!
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from pydantic import BaseModel, validator
from google.cloud import storage
from google.cloud import pubsub_v1
from google.oauth2 import service_account
import tempfile
import os
import json
import logging
import cv2
from typing import Dict, Optional
import uuid
from datetime import datetime
from google.cloud.exceptions import NotFound
import aiohttp
from dotenv import load_dotenv

from inference_json import WasteSegmenter
from vid_analytics_naive import VideoWasteAnalytics
from tracker2 import ConveyorTracker

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
INPUT_BUCKET = os.getenv('INPUT_BUCKET')
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')
TOPIC_NAME = os.getenv('TOPIC_NAME')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_gcp_clients():
    """Initialize GCP clients with proper credentials"""
    credentials = service_account.Credentials.from_service_account_info({
        "type": os.getenv('GCP_TYPE'),
        "project_id": os.getenv('GCP_PROJECT_ID'),
        "private_key_id": os.getenv('GCP_PRIVATE_KEY_ID'),
        "private_key": os.getenv('GCP_PRIVATE_KEY'),
        "client_email": os.getenv('GCP_CLIENT_EMAIL'),
        "client_id": os.getenv('GCP_CLIENT_ID'),
        "auth_uri": os.getenv('GCP_AUTH_URI'),
        "token_uri": os.getenv('GCP_TOKEN_URI'),
        "auth_provider_x509_cert_url": os.getenv('GCP_AUTH_PROVIDER_CERT_URL'),
        "client_x509_cert_url": os.getenv('GCP_CLIENT_CERT_URL'),
        "universe_domain": os.getenv('GCP_UNIVERSE_DOMAIN')
    })
    
    return storage.Client(credentials=credentials), pubsub_v1.PublisherClient(credentials=credentials)

storage_client, publisher = initialize_gcp_clients()
TOPIC_PATH = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

class BucketClient:
    """Client for handling Google Cloud Storage operations"""
    
    def __init__(self, upload_path: str, is_private: bool = False):
        required_vars = [
            "GOOGLE_CLOUD_PROJECT_ID",
            "GOOGLE_CLOUD_CLIENT_EMAIL",
            "GOOGLE_CLOUD_PRIVATE_KEY",
            "GOOGLE_CLOUD_PRIVATE_KEY_ID",
            "GOOGLE_CLOUD_CLIENT_ID",
            "GOOGLE_CLOUD_CLIENT_CERT_URL"
        ]
        
        if not is_private:
            required_vars.append("GOOGLE_CLOUD_PUBLIC_BUCKET")
        else:
            required_vars.append("GOOGLE_CLOUD_PRIVATE_BUCKET")

        credentials = service_account.Credentials.from_service_account_info({
            "type": os.getenv('GCP_TYPE'),
            "project_id": os.getenv('GCP_PROJECT_ID'),
            "private_key_id": os.getenv('GCP_PRIVATE_KEY_ID'),
            "private_key": os.getenv('GCP_PRIVATE_KEY'),
            "client_email": os.getenv('GCP_CLIENT_EMAIL'),
            "client_id": os.getenv('GCP_CLIENT_ID'),
            "auth_uri": os.getenv('GCP_AUTH_URI'),
            "token_uri": os.getenv('GCP_TOKEN_URI'),
            "auth_provider_x509_cert_url": os.getenv('GCP_AUTH_PROVIDER_CERT_URL'),
            "client_x509_cert_url": os.getenv('GCP_CLIENT_CERT_URL'),
            "universe_domain": os.getenv('GCP_UNIVERSE_DOMAIN')
        })
        
        self.client = storage.Client(
            project=os.getenv('GCP_PROJECT_ID'),
            credentials=credentials
        )
        
        self.bucket_name = ("" 
                          if is_private 
                          else os.getenv('INPUT_BUCKET'))
        self.upload_path = upload_path

    async def upload_file(self, file_buffer: bytes, filename: str) -> str:
        """Upload a file to Google Cloud Storage"""
        blob = self.client.bucket(self.bucket_name).blob(f"{self.upload_path}{filename}")
        content_type = "video/mp4" if filename.lower().endswith(('.mp4', '.mov', '.avi')) else "application/octet-stream"
        blob.upload_from_string(file_buffer, content_type=content_type)
        public_url = f"https://storage.googleapis.com/{self.bucket_name}/{self.upload_path}{filename}"
        return public_url

    async def delete_file(self, filename: str) -> None:
        """Delete a file from Google Cloud Storage"""
        blob = self.client.bucket(self.bucket_name).blob(f"{self.upload_path}{filename}")
        try:
            blob.delete()
        except NotFound as e:
            raise ValueError(f"Failed to delete file ({filename}): {str(e)}")

    async def get_file(self, filename: str) -> bytes:
        """Get a file from Google Cloud Storage"""
        blob = self.client.bucket(self.bucket_name).blob(f"{self.upload_path}{filename}")
        try:
            return blob.download_as_bytes()
        except NotFound as e:
            raise ValueError(f"Failed to get file ({filename}): {str(e)}")

project_client = BucketClient("projects/")
video_client = BucketClient("videos/")

class VideoRequest(BaseModel):
    """Request model for video processing"""
    video_gcs_path: str
    callback_url: Optional[str] = None

class ProcessingStatus(BaseModel):
    """Response model for processing status"""
    job_id: str
    status: str = "processing"
    output_video_url: Optional[str] = None
    analytics_url: Optional[str] = None
    error_message: Optional[str] = None
    
    class Config:
        use_enum_values = True
        extra = "forbid"
        
    @validator('status')
    def validate_status(cls, v):
        allowed_statuses = {"processing", "completed", "failed"}
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}")
        return v

def check_video_format(video_path: str) -> bool:
    """Validate video format and readability"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
        cap.release()
        return True
    except Exception:
        return False

async def send_callback(callback_url: str, status_message: dict):
    """Send processing status to callback URL"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(callback_url, json=status_message) as response:
                if response.status >= 400:
                    logger.error(f"Callback failed: {await response.text()}")
    except Exception as e:
        logger.error(f"Error sending callback: {str(e)}")

async def process_video(video_gcs_path: str, job_id: str):
    """Process video from GCS and upload results"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_bucket_name = video_gcs_path.split("/")[2]
            input_blob_name = "/".join(video_gcs_path.split("/")[3:])
            bucket = storage_client.bucket(input_bucket_name)
            blob = bucket.blob(input_blob_name)
            
            local_video_path = os.path.join(temp_dir, "input_video.mp4")
            blob.download_to_filename(local_video_path)
            
            vidcap = cv2.VideoCapture(local_video_path)
            if not vidcap.isOpened():
                raise RuntimeError("Failed to open video file")
            
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            segmenter = WasteSegmenter(
                config_path="zerowaste_config.yaml",
                weights_path="model_final.pth"
            )
            tracker = ConveyorTracker()
            video_analytics = VideoWasteAnalytics(fps, width, height)
            
            output_video_path = os.path.join(temp_dir, "processed_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = vidcap.read()
                if not ret:
                    break
                
                pred, colored_mask = segmenter.predict(frame)
                if pred is None:
                    continue
                
                tracks = tracker.process_frame(pred)
                video_analytics.process_tracks(
                    tracker.get_active_tracks(),
                    tracker.get_completed_tracks()
                )
                
                alpha = 0.6
                vis_frame = cv2.addWeighted(frame, alpha, colored_mask, 1-alpha, 0)
                out.write(vis_frame)
            
            vidcap.release()
            out.release()
            
            analytics = video_analytics.generate_summary()
            analytics_path = os.path.join(temp_dir, "analytics.json")
            with open(analytics_path, 'w') as f:
                json.dump(analytics, f, indent=4)
            
            output_bucket = storage_client.bucket(OUTPUT_BUCKET)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            video_blob_name = f"{job_id}/{timestamp}_processed.mp4"
            video_blob = output_bucket.blob(video_blob_name)
            video_blob.upload_from_filename(output_video_path)
            
            analytics_blob_name = f"{job_id}/{timestamp}_analytics.json"
            analytics_blob = output_bucket.blob(analytics_blob_name)
            analytics_blob.upload_from_filename(analytics_path)
            
            video_url = video_blob.generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            
            analytics_url = analytics_blob.generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            
            status_message = {
                "job_id": job_id,
                "status": "completed",
                "output_video_url": video_url,
                "analytics_url": analytics_url
            }
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        status_message = {
            "job_id": job_id,
            "status": "failed",
            "error_message": str(e)
        }
    
    publisher.publish(
        TOPIC_PATH,
        data=json.dumps(status_message).encode("utf-8")
    )

app = FastAPI(title="Waste Analysis API")

@app.post("/process-video", response_model=ProcessingStatus)
async def start_video_processing(
    request: VideoRequest,
    background_tasks: BackgroundTasks
):
    """Start asynchronous video processing"""
    try:
        job_id = str(uuid.uuid4())
        
        background_tasks.add_task(
            process_video,
            request.video_gcs_path,
            job_id
        )
        
        return ProcessingStatus(
            job_id=job_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get current status of video processing job"""
    try:
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)
        blobs = list(output_bucket.list_blobs(prefix=f"{job_id}/"))
        
        if not blobs:
            return ProcessingStatus(
                job_id=job_id,
                status="processing"
            )
        
        video_blob = next(
            (b for b in blobs if b.name.endswith("processed.mp4")),
            None
        )
        analytics_blob = next(
            (b for b in blobs if b.name.endswith("analytics.json")),
            None
        )
        
        if video_blob and analytics_blob:
            video_url = video_blob.generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            analytics_url = analytics_blob.generate_signed_url(
                version="v4",
                expiration=3600,
                method="GET"
            )
            
            return ProcessingStatus(
                job_id=job_id,
                status="completed",
                output_video_url=video_url,
                analytics_url=analytics_url
            )
            
        return ProcessingStatus(
            job_id=job_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )