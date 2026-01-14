"""
Clip a video from GCS into segments defined in a JSON file and upload clips to GCS.

Usage example:
python clip_from_json.py \
    --video_gcs gs://egocentric-main/Egocentric-Clips/Clip-1/clip-1.mp4 \
    --json_gcs gs://egocentric-main/results_egocentric/Clip-1_tasks.json \
    --clips_gcs_prefix gs://egocentric-main/Sessions/Session-1 \
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
    """Download file from GCS using python client"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_uri} -> {local_path}")


def gcs_upload(local_path: str, gcs_uri: str):
    """Upload file to GCS using python client"""
    client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> {gcs_uri}")


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


# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_gcs", required=True, help="GCS URL of the video to clip")
    parser.add_argument("--json_gcs", required=True, help="GCS URL of the JSON defining segments")
    parser.add_argument("--clips_gcs_prefix", required=True, help="GCS path prefix to upload clips")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        local_video = os.path.join(td, "video.mp4")
        gcs_download(args.video_gcs, local_video)

        local_json = os.path.join(td, "segments.json")
        gcs_download(args.json_gcs, local_json)

        with open(local_json, "r", encoding="utf-8") as f:
            segments: List[dict] = json.load(f)

        for idx, seg in enumerate(segments, start=1):
            start = seg["start_time"]
            end = seg["end_time"]
            clip_name = f"shot_{idx:03d}.mp4"
            local_clip = os.path.join(td, clip_name)
            ffmpeg_clip(local_video, start, end, local_clip)

            gcs_clip_path = f"{args.clips_gcs_prefix}/{clip_name}"
            gcs_upload(local_clip, gcs_clip_path)
            print(f"[OK] Uploaded {clip_name} -> {gcs_clip_path}")

    print("\nAll clips processed and uploaded.")


if __name__ == "__main__":
    main()
