from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import tempfile
import os
from typing import Dict
from urllib.parse import urlparse

class GCPLink(BaseModel):
    url: str

app = FastAPI()

def download_with_wget(url: str) -> str:
    try:
        filename = os.path.basename(urlparse(url).path)
        
        subprocess.run(['wget', '-q', url], check=True)
        return filename
    
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

def process_file(file_path: str) -> Dict:
    try:
        result = {
            "status": "success",
            "processed_file": file_path,
            "message": "File processed successfully"
        }
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/process-gcp-file")
async def process_gcp_file(gcp_link):
    try:
        file_path = download_with_wget(gcp_link)
        result = process_file(file_path)
        return result