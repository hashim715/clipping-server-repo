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

def process_session_folder(session_folder: str, skip_existing: bool, videos_to_flip: list, output_suffix: str):
    """Process a single session folder"""
    session_name = session_folder.split('/')[-1]
    print(f"{'='*80}")
    print(f"{session_name}")
    print(f"{'='*80}")
    
    # List all shot folders
    shot_folders = list_gcs_folders(session_folder)
    
    if not shot_folders:
        print(f"  No shot folders found\n")
        return 0, 0, 0
    
    print(f"Found {len(shot_folders)} shots\n")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for shot_folder in sorted(shot_folders):
        shot_name = shot_folder.split('/')[-1]
        print(f"--- {shot_name} ---")
        
        for video_name in videos_to_flip:
            video_gcs_path = f"{shot_folder}/{video_name}"
            output_name = video_name.replace(".mp4", f"_{output_suffix}.mp4")
            output_gcs_path = f"{shot_folder}/{output_name}"
            
            # Check if original video exists
            if not gcs_file_exists(video_gcs_path):
                print(f"  ⚠ {video_name} not found")
                continue
            
            # Check if copy already exists
            if skip_existing and gcs_file_exists(output_gcs_path):
                print(f"  ✓ {video_name} (skipped - {output_suffix} exists)")
                skipped += 1
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
                    processed += 1
                    
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                errors += 1
        
        print()  # Blank line between shots
    
    return processed, skipped, errors

def flip_wrist_videos(
    gcs_path: str, 
    skip_existing: bool = True, 
    specific_session: str = None,
    output_suffix: str = "Copy",
    exclude_sessions: list = None
):
    """
    Flip Left_Wrist.mp4 and Right_Wrist.mp4 videos by 180 degrees.
    
    Args:
        gcs_path: GCS path to Sessions folder
        skip_existing: Skip if output already exists
        specific_session: Specific session name to process (e.g., "Session-1")
        output_suffix: Suffix for output files (e.g., "Flipped" or "Copy")
        exclude_sessions: List of sessions to exclude (e.g., ["Session-1"])
    """
    setup_gcp_auth()
    
    print(f"\n{'='*80}")
    if specific_session:
        print(f"Flipping wrist videos in: {gcs_path}/{specific_session}")
    else:
        print(f"Flipping wrist videos in: {gcs_path}")
        if exclude_sessions:
            print(f"Excluding: {', '.join(exclude_sessions)}")
    print(f"Output suffix: {output_suffix}")
    print(f"{'='*80}\n")
    
    videos_to_flip = ["Left_Wrist.mp4", "Right_Wrist.mp4"]
    
    total_videos = 0
    processed_videos = 0
    skipped_videos = 0
    error_videos = 0
    
    # Determine which sessions to process
    if specific_session:
        # Process only the specific session
        session_folders = [f"{gcs_path}/{specific_session}"]
        print(f"Processing specific session: {specific_session}\n")
    else:
        # Process all sessions
        session_folders = list_gcs_folders(gcs_path)
        if not session_folders:
            print("No session folders found!")
            return
        
        # Filter out excluded sessions
        if exclude_sessions:
            original_count = len(session_folders)
            session_folders = [
                sf for sf in session_folders 
                if sf.split('/')[-1] not in exclude_sessions
            ]
            excluded_count = original_count - len(session_folders)
            print(f"Found {original_count} session folders ({excluded_count} excluded)\n")
        else:
            print(f"Found {len(session_folders)} session folders\n")
    
    # Process each session
    for session_folder in sorted(session_folders):
        # Count videos in this session
        shot_folders = list_gcs_folders(session_folder)
        session_video_count = len(shot_folders) * len(videos_to_flip)
        total_videos += session_video_count
        
        # Process the session
        p, s, e = process_session_folder(session_folder, skip_existing, videos_to_flip, output_suffix)
        processed_videos += p
        skipped_videos += s
        error_videos += e
    
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
        "--session",
        type=str,
        help="Specific session to process (e.g., Session-1). If not provided, processes all sessions except excluded ones."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="Copy",
        help="Output file suffix (default: Copy). Use 'Flipped' for testing. Results in Left_Wrist_{suffix}.mp4"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["Session-1"],
        help="Sessions to exclude when processing all (default: Session-1). Use --exclude to override."
    )
    parser.add_argument(
        "--no-exclude",
        action="store_true",
        help="Process all sessions including Session-1"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process videos even if output already exists"
    )
    
    args = parser.parse_args()
    
    # Handle exclusions
    exclude_sessions = None if args.no_exclude else args.exclude
    
    try:
        flip_wrist_videos(
            gcs_path=args.sessions_gcs_prefix,
            skip_existing=not args.no_skip_existing,
            specific_session=args.session,
            output_suffix=args.suffix,
            exclude_sessions=exclude_sessions
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())