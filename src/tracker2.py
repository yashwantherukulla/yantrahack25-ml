import cv2
import numpy as np
import json
import time
import logging
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from inference_json import WasteSegmenter

@dataclass
class Blob:
    """Single detected object in a frame"""
    contour: np.ndarray
    centroid: Tuple[int, int]
    area: float
    material_type: str
    frame_number: int
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    hu_moments: np.ndarray

@dataclass
class TrackedObject:
    """Object being tracked across frames"""
    id: int
    material_type: str
    first_seen: int
    last_seen: int
    positions: List[Tuple[int, int]] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    confidence: float = 1.0

class ConveyorTracker:
    def __init__(self, 
                 max_vertical_movement: int = 50,
                 min_area: int = 100,
                 max_area_change: float = 0.3,
                 match_threshold: float = 0.6):
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_number = 0
        self.class_names = ["background", 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic']
        
        self.max_vertical_movement = max_vertical_movement
        self.min_area = min_area
        self.max_area_change = max_area_change
        self.match_threshold = match_threshold
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ConveyorTracker")

    def extract_blobs(self, mask: np.ndarray) -> Dict[str, List[Blob]]:
        """Extract blobs for each material type from segmentation mask"""
        blobs_by_material = defaultdict(list)
        
        for class_idx, material in enumerate(self.class_names[1:], 1):
            binary = (mask == class_idx).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                    
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                x, y, w, h = cv2.boundingRect(contour)
                hu_moments = cv2.HuMoments(M).flatten()
                
                blob = Blob(
                    contour=contour,
                    centroid=(cx, cy),
                    area=area,
                    material_type=material,
                    frame_number=self.frame_number,
                    bounding_box=(x, y, w, h),
                    hu_moments=hu_moments
                )
                blobs_by_material[material].append(blob)
        
        return blobs_by_material

    def calculate_similarity(self, track: TrackedObject, blob: Blob) -> float:
        """Calculate similarity score between tracked object and new blob"""
        if track.material_type != blob.material_type:
            return 0.0
            
        last_pos = track.positions[-1]
        
        vert_diff = abs(last_pos[1] - blob.centroid[1])
        if vert_diff > self.max_vertical_movement:
            return 0.0
            
        if blob.centroid[0] <= last_pos[0]:
            return 0.0
            
        area_ratio = min(track.areas[-1], blob.area) / max(track.areas[-1], blob.area)
        if abs(1 - area_ratio) > self.max_area_change:
            return 0.0
            
        vert_score = 1 - (vert_diff / self.max_vertical_movement)
        area_score = area_ratio
        
        similarity = 0.4 * vert_score + 0.6 * area_score
        
        return similarity

    def update_tracks(self, blobs_by_material: Dict[str, List[Blob]]):
        """Update tracking state with new frame data"""
        matched_blobs = set()
        
        for track_id, track in list(self.tracks.items()):
            if track.last_seen < self.frame_number - 1:
                continue
                
            candidates = blobs_by_material[track.material_type]
            
            similarities = []
            for i, blob in enumerate(candidates):
                if i not in matched_blobs:
                    similarity = self.calculate_similarity(track, blob)
                    if similarity >= self.match_threshold:
                        similarities.append((i, blob, similarity))
            
            if similarities:
                similarities.sort(key=lambda x: x[2], reverse=True)
                best_idx, best_blob, best_similarity = similarities[0]
                matched_blobs.add(best_idx)
                
                track.positions.append(best_blob.centroid)
                track.areas.append(best_blob.area)
                track.last_seen = self.frame_number
                track.confidence = best_similarity
        
        for material, blobs in blobs_by_material.items():
            for i, blob in enumerate(blobs):
                if i not in matched_blobs:
                    if blob.centroid[0] < 100:
                        track = TrackedObject(
                            id=self.next_id,
                            material_type=material,
                            first_seen=self.frame_number,
                            last_seen=self.frame_number,
                            positions=[blob.centroid],
                            areas=[blob.area],
                            states=['active']
                        )
                        self.tracks[self.next_id] = track
                        self.next_id += 1

    def process_frame(self, segmentation_mask: np.ndarray) -> Dict[int, TrackedObject]:
        """Process a new frame and update tracking"""
        self.frame_number += 1
        blobs_by_material = self.extract_blobs(segmentation_mask)
        self.update_tracks(blobs_by_material)
        return self.tracks

    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Return currently active tracks"""
        return {id: track for id, track in self.tracks.items() 
                if track.last_seen >= self.frame_number - 1}

    def get_completed_tracks(self) -> Dict[int, TrackedObject]:
        """Return tracks that have left the frame"""
        return {id: track for id, track in self.tracks.items()
                if track.last_seen < self.frame_number - 1}

def draw_tracks(frame, active_tracks, completed_tracks, frame_count):
    """Draw tracking visualization on frame"""
    vis_frame = frame.copy()
    
    color_map = {
        'rigid_plastic': (255, 0, 0), # R
        'cardboard': (0, 255, 0), # G
        'metal': (0, 0, 255), # B
        'soft_plastic': (255, 0, 255) # P
    }
    
    material_counts = {material: 0 for material in color_map.keys()}
    
    all_tracks = {**active_tracks, **completed_tracks}
    for track_id, track in all_tracks.items():
        if len(track.positions) < 2:
            continue
            
        if track.last_seen >= frame_count - 5:
            material_counts[track.material_type] += 1
            
            points = track.positions[-10:]
            for i in range(1, len(points)):
                pt1 = tuple(map(int, points[i-1]))
                pt2 = tuple(map(int, points[i]))
                color = color_map[track.material_type]
                cv2.line(vis_frame, pt1, pt2, color, 2)
            

            current_pos = track.positions[-1]
            cv2.circle(vis_frame, (int(current_pos[0]), int(current_pos[1])), 
                      5, color_map[track.material_type], -1)
            
            label = f"ID:{track_id} ({track.material_type})"
            cv2.putText(vis_frame, label, 
                       (int(current_pos[0])-10, int(current_pos[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[track.material_type], 2)

    legend_start_y = 60
    for i, (material, color) in enumerate(color_map.items()):
        cv2.rectangle(vis_frame, (10, legend_start_y + i*30), 
                     (30, legend_start_y + i*30 + 20), color, -1)
        count_text = f"{material}: {material_counts[material]}"
        cv2.putText(vis_frame, count_text, 
                   (40, legend_start_y + i*30 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    total_count = sum(material_counts.values())
    cv2.putText(vis_frame, f"Total Objects: {total_count}", 
                (10, legend_start_y + len(color_map)*30 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    return vis_frame

def update_tracks(self, blobs_by_material: Dict[str, List[Blob]]):
    """Update tracking state with new frame data"""
    matched_track_ids = set()
    matched_blobs = {material: set() for material in self.class_names[1:]}
    
    for track_id, track in list(self.tracks.items()):
        if track.last_seen < self.frame_number - 5:
            track.states.append('inactive')
            continue
            
        candidates = blobs_by_material[track.material_type]
        best_match = None
        best_similarity = 0
        best_idx = None
        
        for i, blob in enumerate(candidates):
            if i not in matched_blobs[track.material_type]:
                similarity = self.calculate_similarity(track, blob)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = blob
                    best_idx = i
        
        if best_similarity >= self.match_threshold:
            matched_track_ids.add(track_id)
            matched_blobs[track.material_type].add(best_idx)
            
            track.positions.append(best_match.centroid)
            track.areas.append(best_match.area)
            track.last_seen = self.frame_number
            track.confidence = best_similarity
            track.states.append('active')
        else:
            track.states.append('lost')
    
    for material, blobs in blobs_by_material.items():
        for i, blob in enumerate(blobs):
            if i not in matched_blobs[material]:
                if blob.centroid[0] < 100:
                    track = TrackedObject(
                        id=self.next_id,
                        material_type=material,
                        first_seen=self.frame_number,
                        last_seen=self.frame_number,
                        positions=[blob.centroid],
                        areas=[blob.area],
                        states=['active']
                    )
                    self.tracks[self.next_id] = track
                    self.next_id += 1

if __name__ == "__main__":
    required_files = {
        'config': "zerowaste_config.yaml",
        'weights': "model_final.pth",
        'input_video': "raw.mp4"
    }
    
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required {name} file not found: {path}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("ConveyorTracker")
    
    try:
        segmenter = WasteSegmenter(required_files['config'], required_files['weights'])
        tracker = ConveyorTracker()
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}")
        raise
    
    vidcap = cv2.VideoCapture(required_files['input_video'])
    if not vidcap.isOpened():
        raise RuntimeError("Failed to open input video")
    
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_dir, "segmentation_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    stats = {
        'total_objects': 0,
        'objects_by_material': defaultdict(int),
        'tracked_objects': []
    }
    
    start = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = vidcap.read()
            if not ret or frame_count == 11:
                break
            
            frame_count += 1
            logger.info(f"Processing frame {frame_count}/{total_frames}")
            
            pred, colored_mask = segmenter.predict(frame)
            if pred is None or colored_mask is None:
                logger.warning(f"Segmentation failed for frame {frame_count}")
                continue
            
            tracks = tracker.process_frame(pred)
            completed_tracks = tracker.get_completed_tracks()
            
            vis_frame = frame.copy()
            alpha = 0.6 
            vis_frame = cv2.addWeighted(vis_frame, alpha, colored_mask, 1-alpha, 0)
            
            out.write(vis_frame)
            
            for track_id, track in completed_tracks.items():
                if track_id not in [t['id'] for t in stats['tracked_objects']]:
                    stats['total_objects'] += 1
                    stats['objects_by_material'][track.material_type] += 1
                    
                    track_info = {
                        'id': track_id,
                        'material_type': track.material_type,
                        'first_seen': track.first_seen,
                        'last_seen': track.last_seen,
                        'confidence': track.confidence,
                        'path_length': len(track.positions)
                    }
                    stats['tracked_objects'].append(track_info)
    
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    
    finally:
        stats_file = os.path.join(output_dir, "tracking_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.info(f"Tracking statistics saved to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {str(e)}")
        
        vidcap.release()
        out.release()
        
        finish = time.time()
        logger.info(f"Total execution time: {finish - start:.2f} seconds")
        logger.info(f"Processed {frame_count} frames")
        logger.info(f"Output video saved as {output_video_path}")
        logger.info(f"Total objects tracked: {stats['total_objects']}")
        for material, count in stats['objects_by_material'].items():
            logger.info(f"  {material}: {count} objects")