"""
Clip a video from GCS into segments defined in a JSON file and upload clips to GCS.

Usage example:
python clip_from_json.py \
    --video_gcs gs://egocentric-main/Egocentric-Clips/Clip-1/clip-1.mp4 \
    --json_gcs gs://egocentric-main/results_egocentric/Clip-1_tasks.json \
    --clips_gcs_prefix gs://egocentric-main/Sessions/Session-1 \
    --clip_name Left_Wrist \
    --flip \
    --overwrite
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List
from google.cloud import storage

load_dotenv()

# --------------------------
# GCP authentication
# --------------------------
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


# --------------------------
# Helpers
# --------------------------
def run_cmd(cmd: list, timeout: int = None, check: bool = True):
    """Run a shell command (used for ffmpeg)"""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def gcs_download(gcs_uri: str, local_path: str):
    """Download file from GCS using Python client"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def gcs_upload(local_path: str, gcs_uri: str):
    """Upload file to GCS using Python client"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {gcs_uri}")


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
    run_cmd(cmd)
    print(f"Rotated video 180°: {input_video} -> {output_video}")


def ffmpeg_clip(local_video: str, start: str, end: str, output_path: str):
    """Clip a video segment using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", local_video,
        "-ss", start,
        "-to", end,
        "-c", "copy",
        output_path
    ]
    run_cmd(cmd)


def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = run_cmd(cmd)
    data = json.loads(result.stdout)
    
    # Find video stream
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        raise RuntimeError("No video stream found")
    
    # Extract metadata
    duration = float(data["format"]["duration"])
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    
    # Parse fps
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split("/"))
    fps = round(num / den, 2)
    
    codec = video_stream.get("codec_name", "unknown")
    size_bytes = int(data["format"]["size"])
    
    return {
        "duration_sec": round(duration, 6),
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "size_bytes": size_bytes
    }


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_gcs", required=True, help="GCS URL of the video to clip")
    parser.add_argument("--json_gcs", required=True, help="GCS URL of the JSON defining segments")
    parser.add_argument("--clips_gcs_prefix", required=True, help="GCS path prefix to upload clips")
    parser.add_argument("--clip_name", required=True, help="Name for the clip (e.g., Left_Wrist, Right_Wrist, Clip)")
    parser.add_argument("--flip", action="store_true", help="Rotate video 180 degrees before clipping")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Check if this is the regular clip (not wrist videos)
    is_regular_clip = args.clip_name.lower() == "clip"

    with tempfile.TemporaryDirectory() as td:
        # Download main video
        local_video = os.path.join(td, "video.mp4")
        gcs_download(args.video_gcs, local_video)

        # Rotate video 180° if requested
        if args.flip:
            rotated_video = os.path.join(td, "video_rotated.mp4")
            ffmpeg_rotate_180(local_video, rotated_video)
            local_video = rotated_video

        # Download segments JSON
        local_json = os.path.join(td, "segments.json")
        gcs_download(args.json_gcs, local_json)

        with open(local_json, "r", encoding="utf-8") as f:
            segments: List[dict] = json.load(f)

        # Process each segment
        for idx, seg in enumerate(segments, start=1):
            shot_folder = f"shot_{idx:02d}"
            local_folder = os.path.join(td, shot_folder)
            os.makedirs(local_folder, exist_ok=True)

            clip_filename = f"{args.clip_name}.mp4"
            local_clip = os.path.join(local_folder, clip_filename)
            ffmpeg_clip(local_video, seg["start_time"], seg["end_time"], local_clip)

            # Upload video clip
            gcs_clip_path = f"{args.clips_gcs_prefix}/{shot_folder}/{clip_filename}"
            gcs_upload(local_clip, gcs_clip_path)
            print(f"[OK] Uploaded {shot_folder}/{clip_filename} -> {gcs_clip_path}")

            # Generate and upload metadata only for regular clips
            # Generate and upload metadata only for regular clips
            if is_regular_clip:
                metadata = get_video_metadata(local_clip)
                metadata["codec"] = "H265"  # Override codec
                metadata["task_description"] = seg.get("description", "")
                
                metadata_filename = "metadata.json"
                local_metadata = os.path.join(local_folder, metadata_filename)
                
                with open(local_metadata, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                
                gcs_metadata_path = f"{args.clips_gcs_prefix}/{shot_folder}/{metadata_filename}"
                gcs_upload(local_metadata, gcs_metadata_path)
                print(f"[OK] Uploaded {shot_folder}/{metadata_filename} -> {gcs_metadata_path}")

    flip_status = "with 180° rotation" if args.flip else "without rotation"
    print(f"\nAll clips processed and uploaded for {args.clip_name} ({flip_status}).")


if __name__ == "__main__":
    main()