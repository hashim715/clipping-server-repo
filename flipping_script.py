#!/usr/bin/env python3
"""
Flip wrist videos by 180 degrees and replace the originals in GCS.
Can process a specific video file or entire session folders.
By default, processes all sessions except Session-1.
"""

import os
import subprocess
import tempfile
import argparse
from pathlib import Path
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

def gcs_download(gcs_uri: str, local_path: str):
    """Download file from GCS"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"  Downloaded {gcs_uri.split('/')[-1]}")

def gcs_upload(local_path: str, gcs_uri: str):
    """Upload file to GCS (replaces existing)"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"  Replaced {gcs_uri.split('/')[-1]}")

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

def is_video_file(path: str) -> bool:
    """Check if the path points to a video file"""
    return path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

def list_gcs_folders(gcs_prefix: str):
    """List all folders under a GCS prefix"""
    client = storage.Client()
    bucket_name, prefix = gcs_prefix.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    
    if not prefix.endswith('/'):
        prefix += '/'
    
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    list(blobs)
    
    folders = [f"gs://{bucket_name}/{p.rstrip('/')}" for p in blobs.prefixes]
    return folders

def ffmpeg_rotate_180(input_video: str, output_video: str):
    """Rotate video 180 degrees using ffmpeg with same compression as original"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", "hflip,vflip",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "copy",
        output_video
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

def process_single_video(video_gcs_path: str):
    """Process a single video file"""
    print(f"\n{'='*80}")
    print(f"Processing single video: {video_gcs_path}")
    print(f"{'='*80}\n")
    
    if not gcs_file_exists(video_gcs_path):
        print(f"❌ Video not found: {video_gcs_path}")
        return 0, 1
    
    try:
        print(f"Processing {video_gcs_path.split('/')[-1]}...")
        
        with tempfile.TemporaryDirectory() as td:
            # Download original
            video_name = video_gcs_path.split('/')[-1]
            local_input = os.path.join(td, video_name)
            gcs_download(video_gcs_path, local_input)
            
            # Flip video
            local_output = os.path.join(td, f"flipped_{video_name}")
            print(f"  Rotating 180°...")
            ffmpeg_rotate_180(local_input, local_output)
            
            # Replace original with flipped version
            gcs_upload(local_output, video_gcs_path)
            
            print(f"  ✓ Complete!\n")
            return 1, 0
            
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        return 0, 1

def process_session_folder(session_folder: str, videos_to_flip: list):
    """Process a single session folder"""
    session_name = session_folder.split('/')[-1]
    print(f"\n{'='*80}")
    print(f"{session_name}")
    print(f"{'='*80}")
    
    # List all shot folders
    shot_folders = list_gcs_folders(session_folder)
    
    if not shot_folders:
        print(f"  No shot folders found")
        return 0, 0
    
    print(f"Found {len(shot_folders)} shots\n")
    
    processed = 0
    errors = 0
    
    for shot_folder in sorted(shot_folders):
        shot_name = shot_folder.split('/')[-1]
        print(f"--- {shot_name} ---")
        
        for video_name in videos_to_flip:
            video_gcs_path = f"{shot_folder}/{video_name}"
            
            # Check if original video exists
            if not gcs_file_exists(video_gcs_path):
                print(f"  ⚠ {video_name} not found")
                continue
            
            # Process the video
            try:
                print(f"\n  Processing {video_name}...")
                
                with tempfile.TemporaryDirectory() as td:
                    # Download original
                    local_input = os.path.join(td, video_name)
                    gcs_download(video_gcs_path, local_input)
                    
                    # Flip video
                    local_output = os.path.join(td, f"flipped_{video_name}")
                    print(f"  Rotating 180°...")
                    ffmpeg_rotate_180(local_input, local_output)
                    
                    # Replace original with flipped version
                    gcs_upload(local_output, video_gcs_path)
                    
                    print(f"  ✓ Complete!\n")
                    processed += 1
                    
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                errors += 1
    
    return processed, errors

def flip_wrist_videos(gcs_path: str, skip_confirmation: bool = False):
    """
    Flip wrist videos by 180 degrees.
    Can process either:
    1. A single video file (e.g., gs://bucket/path/video.mp4)
    2. A Sessions folder (processes all sessions except Session-1)
    
    Args:
        gcs_path: GCS path to video file or Sessions folder
        skip_confirmation: Skip confirmation prompt
    """
    setup_gcp_auth()
    
    # Check if it's a video file or folder
    if is_video_file(gcs_path):
        # Process single video file
        print(f"\n{'='*80}")
        print(f"Mode: Single video file")
        print(f"⚠️  WARNING: This will REPLACE the original file!")
        print(f"{'='*80}\n")
        
        if not skip_confirmation:
            print("⚠️  This will permanently replace the original video with flipped version.")
            response = input("Continue? (yes/no): ").strip().lower()
            if response != "yes":
                print("Aborted.")
                return
        
        processed, errors = process_single_video(gcs_path)
        
        print(f"\n{'='*80}")
        print(f"COMPLETE")
        print(f"{'='*80}")
        print(f"Processed: {processed}")
        print(f"Errors: {errors}")
        print(f"{'='*80}\n")
        
    else:
        # Process Sessions folder
        print(f"\n{'='*80}")
        print(f"Mode: Sessions folder")
        print(f"Flipping wrist videos in: {gcs_path}")
        print(f"Excluding: Session-1")
        print(f"⚠️  WARNING: This will REPLACE original files!")
        print(f"{'='*80}\n")
        
        videos_to_flip = ["Left_Wrist.mp4", "Right_Wrist.mp4"]
        
        # Get all session folders
        session_folders = list_gcs_folders(gcs_path)
        
        if not session_folders:
            print("No session folders found!")
            return
        
        # Filter out Session-1
        original_count = len(session_folders)
        session_folders = [
            sf for sf in session_folders 
            if sf.split('/')[-1] != "Session-1"
        ]
        
        excluded_count = original_count - len(session_folders)
        print(f"Found {original_count} session folders")
        print(f"Processing {len(session_folders)} sessions (excluded {excluded_count})\n")
        
        if not session_folders:
            print("No sessions to process after exclusion!")
            return
        
        # Confirm before proceeding
        if not skip_confirmation:
            print("⚠️  This will permanently replace the original videos with flipped versions.")
            response = input("Continue? (yes/no): ").strip().lower()
            if response != "yes":
                print("Aborted.")
                return
        
        print()
        
        total_videos = 0
        processed_videos = 0
        error_videos = 0
        
        # Process each session
        for session_folder in sorted(session_folders):
            # Count videos in this session
            shot_folders = list_gcs_folders(session_folder)
            session_video_count = len(shot_folders) * len(videos_to_flip)
            total_videos += session_video_count
            
            # Process the session
            p, e = process_session_folder(session_folder, videos_to_flip)
            processed_videos += p
            error_videos += e
        
        print(f"\n{'='*80}")
        print(f"COMPLETE")
        print(f"{'='*80}")
        print(f"Total videos: {total_videos}")
        print(f"Processed: {processed_videos}")
        print(f"Errors: {error_videos}")
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Flip videos by 180 degrees and REPLACE originals. Can process a single video file or entire Sessions folder."
    )
    parser.add_argument(
        "gcs_path",
        type=str,
        help="GCS path to video file (e.g., gs://bucket/path/video.mp4) OR Sessions folder (e.g., gs://bucket/Sessions)"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    try:
        flip_wrist_videos(
            gcs_path=args.gcs_path,
            skip_confirmation=args.yes
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())