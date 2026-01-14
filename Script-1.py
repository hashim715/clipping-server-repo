#!/usr/bin/env python3
"""
Gemini task segment generator for egocentric clips (NO gsutil).

Inputs:
  --video_gcs gs://bucket/path/video.mp4
    OR
  --clip <N>  (uses gs://<bucket>/Egocentric-Clips/Clip-N/clip-N.mp4)

Outputs:
  gs://<bucket>/results_egocentric/Clip-<N>_tasks.json

Auth:
  Uses GOOGLE_APPLICATION_CREDENTIALS or GCP_KEY_FILE (.env)
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage

# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------
load_dotenv()

if os.getenv("GCP_KEY_FILE") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    key_path = Path(os.getenv("GCP_KEY_FILE")).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"GCP key not found: {key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
    print(f"✓ GCP credentials loaded: {key_path}")
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print("✓ Using existing GOOGLE_APPLICATION_CREDENTIALS")
else:
    raise RuntimeError("❌ No GCP credentials configured")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def guess_mime(uri: str) -> str:
    return "video/quicktime" if uri.lower().endswith(".mov") else "video/mp4"


def parse_clip_num(uri: str) -> int:
    m = re.search(r"/Clip-(\d+)/clip-(\d+)\.", uri)
    return int(m.group(1)) if m else 0


def normalize_tasks(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []

    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        if "start_time" in t and "end_time" in t:
            out.append({
                "start_time": str(t["start_time"]).strip(),
                "end_time": str(t["end_time"]).strip(),
                "description": str(t.get("description", "")).strip(),
            })
    return out


# ---------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------
def gcs_download(uri: str, local_path: str):
    client = storage.Client()
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).download_to_filename(local_path)


def gcs_upload(local_path: str, uri: str):
    client = storage.Client()
    bucket_name, blob_path = uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_path)


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------
def make_client(project: str, location: str):
    return genai.Client(vertexai=True, project=project, location=location)


def generate_tasks(
    client: genai.Client,
    model: str,
    prompt: str,
    video_uri: str,
) -> List[Dict[str, str]]:

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(
                file_uri=video_uri,
                mime_type=guess_mime(video_uri),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    tasks = getattr(resp, "parsed", None)

    if tasks is None and hasattr(resp, "text"):
        try:
            tasks = json.loads(resp.text)
        except Exception:
            tasks = []

    return normalize_tasks(tasks)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="global")
    ap.add_argument("--bucket", default="egocentric-main")
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--clip", type=int)
    ap.add_argument("--video_gcs")
    args = ap.parse_args()

    if not args.video_gcs and args.clip is None:
        raise RuntimeError("Provide --video_gcs or --clip")

    prompt = Path(args.prompt_file).read_text().strip()
    client = make_client(args.project, args.location)

    if args.video_gcs:
        video_uri = args.video_gcs
    else:
        video_uri = (
            f"gs://{args.bucket}/Egocentric-Clips/"
            f"Clip-{args.clip}/clip-{args.clip}.mp4"
        )

    clip_num = parse_clip_num(video_uri)
    out_uri = f"gs://{args.bucket}/results_egocentric/Clip-{clip_num}_tasks.json"

    print(f"Processing: {video_uri}")

    tasks = generate_tasks(
        client=client,
        model=args.model,
        prompt=prompt,
        video_uri=video_uri,
    )

    print(f"✓ Tasks generated: {len(tasks)}")

    with tempfile.TemporaryDirectory() as td:
        local_json = os.path.join(td, "tasks.json")
        with open(local_json, "w") as f:
            json.dump(tasks, f, indent=2)

        gcs_upload(local_json, out_uri)

    print(f"✓ Uploaded → {out_uri}")
    print("Done.")


if __name__ == "__main__":
    main()
