from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import requests
import tempfile
from pathlib import Path
from base64 import b64encode
from inference_json import WasteSegmenter
from pydantic import BaseModel


class ImgURL(BaseModel):
    image_url: str

app = FastAPI(
    title="Waste Segmentation API",
    description="API for waste segmentation and analytics",
    version="1.0.0"
)

segmenter = WasteSegmenter(
    config_path="zerowaste_config.yaml",
    weights_path="model_final.pth"
)

@app.post("/analyze-waste/")
async def analyze_waste(request: ImgURL):
    try:
        image_url = request.image_url
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"error": "Failed to download image"}
            )
        print(response)
        
        image_data = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )
        print(image)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "input.jpg"
            output_path = Path(temp_dir) / "output.jpg"
            mask_path = Path(temp_dir) / "mask.jpg"
            
            cv2.imwrite(str(temp_path), image)
            
            segmenter.predict_and_save(str(temp_path), str(output_path), str(mask_path))
            
            analytics = segmenter.send_img_analytics(str(temp_path))
            
            with open(output_path, "rb") as f:
                visualization_bytes = f.read()
            
            with open(mask_path, "rb") as f:
                mask_bytes = f.read()
            print(analytics)
            return JSONResponse(content={
                "analytics": analytics,
                "visualization": b64encode(visualization_bytes).decode(),
                "mask": b64encode(mask_bytes).decode()
            })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Waste Segmentation API is running"}
