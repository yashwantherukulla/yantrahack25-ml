from moviepy.editor import VideoFileClip
import os
from typing import List, Dict, Optional, Callable
import json
from celery import Celery
import time
import asyncio
from celery.result import AsyncResult
from dotenv import load_dotenv

load_dotenv()

def split_video(video_path: str, chunk_duration: int = 30) -> List[str]:
    video = VideoFileClip(video_path)
    duration = video.duration
    chunks = []
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for start in range(0, int(duration), chunk_duration):
        end = min(start + chunk_duration, duration)
        chunk = video.subclip(start, end)
        chunk_path = f"chunks/{base_name}_chunk_{start}_{end}.mp4"
        
        os.makedirs("chunks", exist_ok=True)
        
        chunk.write_videofile(chunk_path)
        chunks.append(chunk_path)
    
    video.close()
    return chunks

celery_app = Celery('video_processor',
                    broker=os.getenv("CELERY_BROKER_URL"),
                    backend=os.getenv("CELERY_RESULT_BACKEND"))

PROCESSING_FUNCTIONS = {}

def register_processing_function(name: str):
    def decorator(func):
        PROCESSING_FUNCTIONS[name] = func
        return func
    return decorator

@celery_app.task
def process_chunk(chunk_path: str, processing_function_name: str) -> Dict:
    if processing_function_name not in PROCESSING_FUNCTIONS:
        raise ValueError(f"Unknown processing function: {processing_function_name}")
    
    processing_function = PROCESSING_FUNCTIONS[processing_function_name]
    processed_video, metrics = processing_function(chunk_path)
    
    return {
        'processed_path': processed_video,
        'metrics': metrics
    }

class ChunkProcessor:
    def __init__(self, processing_function_name: str):
        if processing_function_name not in PROCESSING_FUNCTIONS:
            raise ValueError(f"Unknown processing function: {processing_function_name}")
        self.processing_function_name = processing_function_name
        self.results = []
    
    async def process_video_sequentially(self, video_path: str, chunk_duration: int = 1) -> List[Dict]:
        chunks = split_video(video_path, chunk_duration)
        for chunk_path in chunks:
            task = process_chunk.delay(chunk_path, self.processing_function_name)
            
            while not task.ready():
                print(1)
                await asyncio.sleep(1)
            
            result = task.get()
            self.results.append(result)
            
            os.remove(chunk_path)
            
        print("done")    
        return self.results

@register_processing_function('example_processor')
def example_processing_function(video_path: str) -> tuple:
    print("task x is done")
    processed_path = f"processed_{os.path.basename(video_path)}"
    metrics = {
        "frame_count": 300,
        "processing_time": 5,
        "quality_score": 0.95
    }
    
    return processed_path, metrics

async def main():
    processor = ChunkProcessor('example_processor')
    
    video_path = "testvideo.mp4"
    results = await processor.process_video_sequentially(video_path)
    
    with open("processing_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())