#!/usr/bin/env python3
"""
3D Hand Tracking using MediaPipe
Processes video shots and extracts 3D hand landmarks data.
Supports batch processing of Sessions folders in GCS.
"""

import cv2
import json
import os
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import numpy as np
import urllib.request
import tempfile
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

# --------------------------
# GCP authentication
# --------------------------
def setup_gcp_auth():
    """Setup GCP authentication"""
    if os.getenv("GCP_KEY_FILE") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        gcp_key_path = Path(os.getenv("GCP_KEY_FILE")).resolve()
        if not gcp_key_path.exists():
            raise FileNotFoundError(f"GCP key file not found: {gcp_key_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gcp_key_path)
        print(f"✓ GCP credentials loaded from: {gcp_key_path}")
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print(f"✓ Using existing GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    else:
        raise RuntimeError("❌ No GCP credentials configured")

def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS URI"""
    return path.startswith("gs://")

def gcs_download(gcs_uri: str, local_path: str):
    """Download file from GCS"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")

def gcs_upload(local_path: str, gcs_uri: str):
    """Upload file to GCS"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {gcs_uri}")

def list_gcs_folders(gcs_prefix: str) -> List[str]:
    """List all folders under a GCS prefix"""
    client = storage.Client()
    bucket_name, prefix = gcs_prefix.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    
    # Add trailing slash if not present
    if not prefix.endswith('/'):
        prefix += '/'
    
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    
    # Consume the iterator to get prefixes
    list(blobs)
    
    folders = [f"gs://{bucket_name}/{p.rstrip('/')}" for p in blobs.prefixes]
    return folders

def gcs_file_exists(gcs_uri: str) -> bool:
    """Check if a file exists in GCS"""
    try:
        client = storage.Client()
        bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except:
        return False

# Try to import MediaPipe with better error handling
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_NEW_API = True
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")
    print("\nPlease install MediaPipe using:")
    print("  pip install mediapipe")
    raise

def process_video_for_hand_tracking(
    video_path: str,
    output_gcs_path: str,
    save_visualization: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> Dict:
    """
    Process a video file to extract 3D hand tracking data using MediaPipe.
    
    Args:
        video_path: GCS path to input video file
        output_gcs_path: GCS path where to save the JSON output
        save_visualization: Whether to save a video with hand landmarks visualized
        min_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
        min_tracking_confidence: Minimum confidence for hand tracking (0.0-1.0)
    
    Returns:
        Dictionary containing tracking results and metadata
    """
    print(f"\n{'='*80}")
    print(f"Processing video: {video_path}")
    print(f"{'='*80}")
    
    # Download video from GCS to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video_path = temp_video.name
        gcs_download(video_path, temp_video_path)
    
    try:
        # Open video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames, {duration:.2f}s")
        
        # Download model file if needed
        model_dir = Path.home() / ".mediapipe" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"
        
        if not model_path.exists():
            print("Downloading hand landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Model downloaded to: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model: {e}\n"
                    f"Please manually download the model from:\n"
                    f"{model_url}\n"
                    f"and save it to: {model_path}"
                )
        else:
            print(f"Using existing model: {model_path}")
        
        # Initialize MediaPipe Hands
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Initialize video writer for visualization if needed
        vis_writer = None
        temp_vis_path = None
        if save_visualization:
            temp_vis_path = temp_video_path.replace(".mp4", "_vis.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vis_writer = cv2.VideoWriter(temp_vis_path, fourcc, fps, (width, height))
        
        # Storage for tracking data
        tracking_data = {
            "video_info": {
                "input_file": video_path,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": duration
            },
            "frames": []
        }
        
        frame_count = 0
        hands_detected_count = 0
        
        print("Processing frames...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert BGR to RGB (MediaPipe uses RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame for hand tracking
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int((frame_count / fps) * 1000)
                detection_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Store frame data
                frame_data = {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps if fps > 0 else 0,
                    "hands": []
                }
                
                # Process detected hands
                if detection_result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        # Get hand label
                        hand_label = "Unknown"
                        hand_score = 0.0
                        if detection_result.handedness and hand_idx < len(detection_result.handedness):
                            if len(detection_result.handedness[hand_idx]) > 0:
                                hand_label = detection_result.handedness[hand_idx][0].category_name
                                hand_score = detection_result.handedness[hand_idx][0].score
                        
                        # Extract 3D landmarks
                        landmarks_3d = []
                        landmarks_2d = []
                        
                        for landmark in hand_landmarks:
                            landmarks_3d.append({
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": getattr(landmark, 'visibility', 1.0)
                            })
                            landmarks_2d.append({
                                "x": landmark.x * width,
                                "y": landmark.y * height
                            })
                        
                        hand_data = {
                            "hand_index": hand_idx,
                            "label": hand_label,
                            "confidence": float(hand_score),
                            "landmarks_3d": landmarks_3d,
                            "landmarks_2d": landmarks_2d
                        }
                        
                        frame_data["hands"].append(hand_data)
                        hands_detected_count += 1
                        
                        # Draw hand landmarks on frame for visualization
                        if save_visualization:
                            for landmark in hand_landmarks:
                                x = int(landmark.x * width)
                                y = int(landmark.y * height)
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                            
                            # Draw connections
                            connections = [
                                (0, 1), (1, 2), (2, 3), (3, 4),
                                (0, 5), (5, 6), (6, 7), (7, 8),
                                (0, 9), (9, 10), (10, 11), (11, 12),
                                (0, 13), (13, 14), (14, 15), (15, 16),
                                (0, 17), (17, 18), (18, 19), (19, 20),
                                (5, 9), (9, 13), (13, 17)
                            ]
                            for start_idx, end_idx in connections:
                                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                                    start = hand_landmarks[start_idx]
                                    end = hand_landmarks[end_idx]
                                    cv2.line(
                                        frame,
                                        (int(start.x * width), int(start.y * height)),
                                        (int(end.x * width), int(end.y * height)),
                                        (0, 255, 0),
                                        2
                                    )
                
                tracking_data["frames"].append(frame_data)
                
                # Write visualization frame
                if save_visualization and vis_writer:
                    vis_writer.write(frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    hands_in_frame = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
                    print(f"Frame {frame_count}/{total_frames}: {hands_in_frame} hands detected")
        
        finally:
            # Cleanup
            cap.release()
            hand_landmarker.close()
            if vis_writer:
                vis_writer.release()
        
        # Add summary statistics
        tracking_data["summary"] = {
            "total_frames_processed": frame_count,
            "frames_with_hands": len([f for f in tracking_data["frames"] if f["hands"]]),
            "total_hand_detections": hands_detected_count,
            "average_hands_per_frame": hands_detected_count / frame_count if frame_count > 0 else 0
        }
        
        # Save JSON to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as temp_json:
            temp_json_path = temp_json.name
            json.dump(tracking_data, temp_json, indent=2)
        
        # Upload JSON to GCS
        print(f"\nUploading tracking data to: {output_gcs_path}")
        gcs_upload(temp_json_path, output_gcs_path)
        
        # Upload visualization if created
        if save_visualization and temp_vis_path and os.path.exists(temp_vis_path):
            vis_gcs_path = output_gcs_path.replace("_hand_tracking.json", "_hand_tracking_vis.mp4")
            print(f"Uploading visualization to: {vis_gcs_path}")
            gcs_upload(temp_vis_path, vis_gcs_path)
            os.unlink(temp_vis_path)
        
        # Cleanup temp files
        os.unlink(temp_json_path)
        
        print(f"\n✓ Processing complete!")
        print(f"  - Frames processed: {frame_count}")
        print(f"  - Frames with hands: {tracking_data['summary']['frames_with_hands']}")
        print(f"  - Total hand detections: {hands_detected_count}")
        print(f"  - JSON output: {output_gcs_path}")
        
        return tracking_data
    
    finally:
        # Cleanup temp video
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def process_sessions_folder(
    sessions_gcs_prefix: str,
    save_visualization: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    skip_existing: bool = True
):
    """
    Process all videos in Sessions folder structure.
    
    Args:
        sessions_gcs_prefix: GCS path to Sessions folder (e.g., gs://bucket/Sessions)
        save_visualization: Whether to save visualization videos
        min_detection_confidence: Minimum confidence for hand detection
        min_tracking_confidence: Minimum confidence for hand tracking
        skip_existing: Skip processing if JSON already exists
    """
    setup_gcp_auth()
    
    print(f"\n{'='*80}")
    print(f"Scanning Sessions folder: {sessions_gcs_prefix}")
    print(f"{'='*80}\n")
    
    # List all session folders (Session-1, Session-2, etc.)
    session_folders = list_gcs_folders(sessions_gcs_prefix)
    
    if not session_folders:
        print("No session folders found!")
        return
    
    print(f"Found {len(session_folders)} session folders:")
    for folder in session_folders:
        print(f"  - {folder}")
    
    video_names = ["Clip.mp4", "Left_Wrist.mp4", "Right_Wrist.mp4"]
    
    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    
    for session_folder in session_folders:
        session_name = session_folder.split('/')[-1]
        print(f"\n{'='*80}")
        print(f"Processing {session_name}")
        print(f"{'='*80}")
        
        for video_name in video_names:
            video_gcs_path = f"{session_folder}/{video_name}"
            json_output_name = video_name.replace(".mp4", "_hand_tracking.json")
            json_gcs_path = f"{session_folder}/{json_output_name}"
            
            total_videos += 1
            
            # Check if video exists
            if not gcs_file_exists(video_gcs_path):
                print(f"\n⚠ Video not found: {video_gcs_path}")
                continue
            
            # Check if JSON already exists
            if skip_existing and gcs_file_exists(json_gcs_path):
                print(f"\n✓ Skipping {video_name} (JSON already exists)")
                skipped_videos += 1
                continue
            
            # Process the video
            try:
                process_video_for_hand_tracking(
                    video_path=video_gcs_path,
                    output_gcs_path=json_gcs_path,
                    save_visualization=save_visualization,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                processed_videos += 1
            except Exception as e:
                print(f"\n❌ Error processing {video_gcs_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos found: {total_videos}")
    print(f"Processed: {processed_videos}")
    print(f"Skipped (already exist): {skipped_videos}")
    print(f"{'='*80}\n")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Extract 3D hand tracking data from videos in Sessions folder structure"
    )
    parser.add_argument(
        "sessions_gcs_prefix",
        type=str,
        help="GCS path to Sessions folder (e.g., gs://bucket/Sessions)"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip saving visualization videos"
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand detection (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand tracking (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process videos even if JSON already exists"
    )
    
    args = parser.parse_args()
    
    try:
        process_sessions_folder(
            sessions_gcs_prefix=args.sessions_gcs_prefix,
            save_visualization=not args.no_visualization,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            skip_existing=not args.no_skip_existing
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())