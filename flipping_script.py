#!/usr/bin/env python3
"""
Flip wrist videos by 180 degrees and save as copies in GCS.
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
    """Upload file to GCS"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"  Uploaded {gcs_uri.split('/')[-1]}")

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
    """Rotate video 180 degrees using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", "hflip,vflip",
        "-c:a", "copy",
        output_video
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

def flip_wrist_videos(sessions_gcs_prefix: str, skip_existing: bool = True):
    """
    Flip Left_Wrist.mp4 and Right_Wrist.mp4 videos by 180 degrees.
    Saves as Left_Wrist_Copy.mp4 and Right_Wrist_Copy.mp4.
    
    Args:
        sessions_gcs_prefix: GCS path to Sessions folder
        skip_existing: Skip if copy already exists
    """
    setup_gcp_auth()
    
    print(f"\n{'='*80}")
    print(f"Flipping wrist videos in: {sessions_gcs_prefix}")
    print(f"{'='*80}\n")
    
    # List all session folders
    session_folders = list_gcs_folders(sessions_gcs_prefix)
    
    if not session_folders:
        print("No session folders found!")
        return
    
    print(f"Found {len(session_folders)} session folders\n")
    
    videos_to_flip = ["Left_Wrist.mp4", "Right_Wrist.mp4"]
    
    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    error_videos = 0
    
    for session_folder in sorted(session_folders):
        session_name = session_folder.split('/')[-1]
        print(f"{'='*80}")
        print(f"{session_name}")
        print(f"{'='*80}")
        
        # List all shot folders
        shot_folders = list_gcs_folders(session_folder)
        
        if not shot_folders:
            print(f"  No shot folders found\n")
            continue
        
        print(f"Found {len(shot_folders)} shots\n")
        
        for shot_folder in sorted(shot_folders):
            shot_name = shot_folder.split('/')[-1]
            print(f"--- {shot_name} ---")
            
            for video_name in videos_to_flip:
                video_gcs_path = f"{shot_folder}/{video_name}"
                output_name = video_name.replace(".mp4", "_Copy.mp4")
                output_gcs_path = f"{shot_folder}/{output_name}"
                
                total_videos += 1
                
                # Check if original video exists
                if not gcs_file_exists(video_gcs_path):
                    print(f"  ⚠ {video_name} not found")
                    continue
                
                # Check if copy already exists
                if skip_existing and gcs_file_exists(output_gcs_path):
                    print(f"  ✓ {video_name} (skipped - copy exists)")
                    skipped_videos += 1
                    continue
                
                # Process the video
                try:
                    print(f"\n  Processing {video_name}...")
                    
                    with tempfile.TemporaryDirectory() as td:
                        # Download original
                        local_input = os.path.join(td, video_name)
                        gcs_download(video_gcs_path, local_input)
                        
                        # Flip video
                        local_output = os.path.join(td, output_name)
                        print(f"  Rotating 180°...")
                        ffmpeg_rotate_180(local_input, local_output)
                        
                        # Upload flipped version
                        gcs_upload(local_output, output_gcs_path)
                        
                        print(f"  ✓ Complete!\n")
                        processed_videos += 1
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}\n")
                    error_videos += 1
            
            print()  # Blank line between shots
    
    print(f"{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos: {total_videos}")
    print(f"Processed: {processed_videos}")
    print(f"Skipped: {skipped_videos}")
    print(f"Errors: {error_videos}")
    print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Flip Left_Wrist and Right_Wrist videos by 180 degrees in GCS Sessions folder"
    )
    parser.add_argument(
        "sessions_gcs_prefix",
        type=str,
        help="GCS path to Sessions folder (e.g., gs://bucket/Sessions)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process videos even if copy already exists"
    )
    
    args = parser.parse_args()
    
    try:
        flip_wrist_videos(
            sessions_gcs_prefix=args.sessions_gcs_prefix,
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