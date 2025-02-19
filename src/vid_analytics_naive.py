from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import json
import logging
from collections import defaultdict
from datetime import datetime
from tracker2 import WasteSegmenter, ConveyorTracker
import cv2
import os

@dataclass
class MaterialProperties:
    """Physical properties of different waste materials"""
    density: float  # kg/pixel
    compressibility: float  # factor to account for air gaps
    
    @classmethod
    def get_default_properties(cls) -> Dict[str, 'MaterialProperties']:
        """Get default material properties based on empirical measurements"""
        return {
            "rigid_plastic": cls(0.00002, 0.8),
            "cardboard": cls(0.00001, 0.6),
            "metal": cls(0.00005, 0.9),
            "soft_plastic": cls(0.000008, 0.5)
        }

@dataclass
class ObjectAnalytics:
    """Analytics for a single tracked waste object"""
    object_id: int
    material_type: str
    first_seen: int
    last_seen: int
    mean_area: float
    area_std: float
    estimated_weight: float
    confidence: float
    track_duration: int
    areas: List[float] = field(default_factory=list)

class VideoWasteAnalytics:
    """Analyzes waste composition and throughput from tracked objects in video"""
    
    def __init__(self, fps: float, frame_width: int, frame_height: int):
        """
        Initialize video analytics processor
        
        Args:
            fps: Frames per second of the video
            frame_width: Width of video frames in pixels
            frame_height: Height of video frames in pixels
        """
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        
        self.material_properties = MaterialProperties.get_default_properties()
        self.objects: Dict[int, ObjectAnalytics] = {}
        self.completed_objects: Dict[int, ObjectAnalytics] = {}
        
        self.logger = logging.getLogger("VideoWasteAnalytics")
        
    def update_object(self, track_id: int, track: 'TrackedObject') -> None:
        """
        Update analytics for a tracked object
        
        Args:
            track_id: Unique identifier for the tracked object
            track: TrackedObject instance containing tracking data
        """
        if track_id not in self.objects:
            self.objects[track_id] = ObjectAnalytics(
                object_id=track_id,
                material_type=track.material_type,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                mean_area=0.0,
                area_std=0.0,
                estimated_weight=0.0,
                confidence=track.confidence,
                track_duration=0,
                areas=track.areas
            )
        else:
            obj = self.objects[track_id]
            obj.last_seen = track.last_seen
            obj.areas = track.areas
            obj.confidence = track.confidence
            
        obj = self.objects[track_id]
        obj.mean_area = float(np.mean(obj.areas))
        obj.area_std = float(np.std(obj.areas))
        obj.track_duration = obj.last_seen - obj.first_seen
        
        material_props = self.material_properties[obj.material_type]
        obj.estimated_weight = (
            obj.mean_area * 
            material_props.density * 
            material_props.compressibility
        )
    
    def complete_object(self, track_id: int) -> None:
        """
        Move an object from active to completed tracking
        
        Args:
            track_id: ID of the object to complete
        """
        if track_id in self.objects:
            self.completed_objects[track_id] = self.objects[track_id]
            del self.objects[track_id]
    
    def process_tracks(self, active_tracks: Dict[int, 'TrackedObject'], 
                      completed_tracks: Dict[int, 'TrackedObject']) -> None:
        """
        Process all tracked objects for the current frame
        
        Args:
            active_tracks: Currently tracked objects
            completed_tracks: Objects that are no longer being tracked
        """
        for track_id, track in active_tracks.items():
            self.update_object(track_id, track)
            
        for track_id, track in completed_tracks.items():
            if track_id in self.objects:
                self.complete_object(track_id)
    
    def generate_summary(self) -> Dict:
        """
        Generate comprehensive analytics summary
        
        Returns:
            Dictionary containing full analytics summary
        """
        all_objects = {**self.completed_objects, **self.objects}
        
        material_stats = defaultdict(lambda: {
            'count': 0,
            'total_weight': 0.0,
            'mean_area': [],
            'mean_duration': [],
            'confidence': []
        })
        
        for obj in all_objects.values():
            stats = material_stats[obj.material_type]
            stats['count'] += 1
            stats['total_weight'] += obj.estimated_weight
            stats['mean_area'].append(obj.mean_area)
            stats['mean_duration'].append(obj.track_duration)
            stats['confidence'].append(obj.confidence)
        
        for material, stats in material_stats.items():
            stats['mean_area'] = float(np.mean(stats['mean_area']))
            stats['mean_duration'] = float(np.mean(stats['mean_duration']))
            stats['mean_confidence'] = float(np.mean(stats['confidence']))
            stats['total_weight'] = float(stats['total_weight'])
        
        total_duration_seconds = max(obj.last_seen for obj in all_objects.values()) / self.fps
        total_weight = sum(stats['total_weight'] for stats in material_stats.values())
        
        if total_duration_seconds > 0:
            throughput_per_hour = (total_weight / total_duration_seconds) * 3600
        else:
            throughput_per_hour = 0.0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'video_info': {
                'duration_seconds': float(total_duration_seconds),
                'frame_width': self.frame_width,
                'frame_height': self.frame_height,
                'fps': self.fps
            },
            'material_composition': dict(material_stats),
            'total_stats': {
                'total_objects': len(all_objects),
                'total_weight_kg': float(total_weight),
                'throughput_kg_per_hour': float(throughput_per_hour),
                'active_tracks': len(self.objects),
                'completed_tracks': len(self.completed_objects)
            }
        }
    
    def save_analytics(self, output_path: str) -> None:
        """
        Save analytics summary to JSON file
        
        Args:
            output_path: Path to save the JSON output
        """
        try:
            summary = self.generate_summary()
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=4)
            self.logger.info(f"Analytics saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save analytics: {str(e)}")
            raise

if __name__ == "__main__":
    required_files = {
        'config': "zerowaste_config.yaml",
        'weights': "model_final.pth",
        'input_video': "train_100.mp4"
    }
    tracker = ConveyorTracker()
    vidcap = cv2.VideoCapture(required_files['input_video'])
    video_analytics = VideoWasteAnalytics(
        fps=vidcap.get(cv2.CAP_PROP_FPS),
        frame_width=int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        frame_height=int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    segmenter = WasteSegmenter(required_files['config'], required_files['weights'])
    
    os.makedirs("output/frames", exist_ok=True)
    
    output_fps = 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/output_video.mp4', 
                         fourcc, 
                         output_fps,
                         (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_count = 0
    if not vidcap.isOpened():
        raise RuntimeError("Failed to open input video")
        
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
            
        pred, colored_mask = segmenter.predict(frame)
        tracks = tracker.process_frame(pred)
        
        overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(f"output/frames/frame_{frame_count:04d}.jpg", overlay)
        out.write(overlay)
        
        video_analytics.process_tracks(
            tracker.get_active_tracks(),
            tracker.get_completed_tracks()
        )
        
        frame_count += 1
        
    vidcap.release()
    out.release()
    video_analytics.save_analytics("output/waste_analytics.json")