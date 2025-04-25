# SORTIQ: Waste Segmentation and Analytics

## Overview

SORTIQ is a machine learning-based waste segmentation and analytics solution designed to automate the identification and analysis of waste materials in images and videos. This project aims to facilitate efficient waste management by providing detailed insights into waste composition and throughput.

## Features

- **Waste Segmentation**: Utilizes a deep learning model to segment and classify different types of waste materials in images.
- **Object Detection**: Utilizes a deep learning model to detect types of waste via bounding boxes.
- **Video Analytics**: Analyzes video streams to track waste objects, estimate their weight, and calculate throughput.
- **API Integration**: Provides RESTful APIs for waste segmentation and video processing, enabling seamless integration with other systems.
- **Real-time Processing**: Supports real-time processing of video streams for immediate waste analytics.
- **Detailed Reports**: Generates comprehensive reports on waste composition, weight, and throughput.

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker (for running Celery workers)
- Google Cloud SDK (for GCP integration)

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-repo/sortiq.git
   cd sortiq
   ```

2. **Set Up Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the Model**:

   ```bash
   python src/train_net.py --config-file src/configs/zerowaste_config.yaml --dataroot /path/to/dataset
   ```

5. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add the necessary environment variables for GCP and Celery.

6. **Run the Application**:
   ```bash
   uvicorn src.img_api:app --reload
   ```

## Usage

### Image Segmentation API

- **Endpoint**: `/analyze-waste/`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "image_url": "https://example.com/path/to/image.jpg"
  }
  ```
- **Response**:
  ```json
  {
    "analytics": {
      "image_size": {
        "height": 1080,
        "width": 1920
      },
      "waste_composition": {
        "rigid_plastic": {
          "present": true,
          "pixel_count": 31143,
          "area_percentage": 1.5,
          "estimated_weight_kg": 0.623
        },
        "cardboard": {
          "present": true,
          "pixel_count": 289435,
          "area_percentage": 13.96,
          "estimated_weight_kg": 2.894
        },
        "metal": {
          "present": false,
          "pixel_count": 0,
          "area_percentage": 0.0,
          "estimated_weight_kg": 0.0
        },
        "soft_plastic": {
          "present": true,
          "pixel_count": 105207,
          "area_percentage": 5.07,
          "estimated_weight_kg": 0.842
        }
      },
      "total_stats": {
        "total_weight_kg": 4.359,
        "waste_coverage": 20.53,
        "distribution_evenness": 0.88
      }
    },
    "visualization": "base64_encoded_image",
    "mask": "base64_encoded_mask"
  }
  ```

### Video Processing API

(WIP)

- **Endpoint**: `/process-video`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "video_gcs_path": "gs://bucket-name/path/to/video.mp4",
    "callback_url": "https://example.com/callback"
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "unique-job-id",
    "status": "processing"
  }
  ```

### Check Processing Status

(WIP)

- **Endpoint**: `/status/{job_id}`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "job_id": "unique-job-id",
    "status": "completed",
    "output_video_url": "https://example.com/output_video.mp4",
    "analytics_url": "https://example.com/analytics.json"
  }
  ```
