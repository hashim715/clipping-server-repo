#!/usr/bin/env python3
"""
Gemini task segment generator for egocentric clips (GCS SDK version).

Inputs:
  gs://<bucket>/Egocentric-Clips/Clip-<N>/clip-<N>.mp4

Outputs:
  gs://<bucket>/results_egocentric/Clip-<N>_tasks.json
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

# -------------------------------------------------
# Environment + credentials
# -------------------------------------------------

load_dotenv()

if os.getenv("GCP_KEY_FILE") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    key_path = Path(os.getenv("GCP_KEY_FILE")).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"GCP key file not found: {key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
    print(f"✓ GCP credentials loaded from: {key_path}")
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print(f"✓ Using GOOGLE_APPLICATION_CREDENTIALS")
else:
    raise RuntimeError("❌ No GCP credentials configured")

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI")
    bucket, blob = uri[5:].split("/", 1)
    return bucket, blob


def parse_clip_num(uri: str) -> int:
    m = re.search(r"/Clip-(\d+)/clip-(\d+)\.", uri)
    return int(m.group(1)) if m else 0


def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Path(path).read_text(encoding="utf-8").strip()


def guess_mime(uri: str) -> str:
    return "video/quicktime" if uri.lower().endswith(".mov") else "video/mp4"


def normalize_tasks(tasks: Any) -> List[Dict[str, str]]:
    if not isinstance(tasks, list):
        return []
    out = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        s = t.get("start_time") or t.get("start")
        e = t.get("end_time") or t.get("end")
        d = t.get("description", "")
        if s and e:
            out.append({
                "start_time": str(s).strip(),
                "end_time": str(e).strip(),
                "description": str(d).strip(),
            })
    return out


# -------------------------------------------------
# Gemini
# -------------------------------------------------

def make_genai_client(project: str, location: str) -> genai.Client:
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
            types.Part.from_uri(video_uri, mime_type=guess_mime(video_uri)),
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
            tasks = None

    return normalize_tasks(tasks)


# -------------------------------------------------
# Main
# -------------------------------------------------

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

    prompt = load_prompt(args.prompt_file)
    genai_client = make_genai_client(args.project, args.location)
    storage_client = storage.Client()

    # Resolve video URI
    if args.video_gcs:
        video_uri = args.video_gcs
    elif args.clip is not None:
        video_uri = (
            f"gs://{args.bucket}/Egocentric-Clips/Clip-{args.clip}/clip-{args.clip}.mp4"
        )
    else:
        raise RuntimeError("Provide --video_gcs or --clip")

    print(f"Processing: {video_uri}")

    # Download video
    bucket_name, blob_path = parse_gcs_uri(video_uri)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        raise RuntimeError(f"❌ Video not found or no permission: {video_uri}")

    with tempfile.TemporaryDirectory() as td:
        local_video = os.path.join(td, os.path.basename(blob_path))
        blob.download_to_filename(local_video)
        print(f"⬇ Downloaded video")

        tasks = generate_tasks(
            client=genai_client,
            model=args.model,
            prompt=prompt,
            video_uri=video_uri,
        )

        clip_num = parse_clip_num(video_uri)
        out_blob_path = f"results_egocentric/Clip-{clip_num}_tasks.json"
        out_blob = bucket.blob(out_blob_path)

        local_json = os.path.join(td, "tasks.json")
        with open(local_json, "w") as f:
            json.dump(tasks, f, indent=2)

        out_blob.upload_from_filename(local_json)
        print(f"✓ Uploaded → gs://{args.bucket}/{out_blob_path}")

    print("Done ✅")


if __name__ == "__main__":
    main()
