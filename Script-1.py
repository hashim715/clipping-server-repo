#!/usr/bin/env python3
"""
Gemini task segment generator for egocentric clips.

Inputs (GCS):
  gs://<bucket>/Egocentric-Clips/Clip-<N>/clip-<N>.mp4

Output (GCS):
  gs://<bucket>/results_egocentric/Clip-<N>_tasks.json
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# -------------------------------------------------
# ENV + GCP AUTH
# -------------------------------------------------

load_dotenv()

if os.getenv("GCP_KEY_FILE") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    key_path = Path(os.getenv("GCP_KEY_FILE")).resolve()
    if not key_path.exists():
        raise FileNotFoundError(f"GCP key file not found: {key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)
    print(f"✓ GCP credentials loaded from: {key_path}")
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print(f"✓ Using existing GOOGLE_APPLICATION_CREDENTIALS")
else:
    print("⚠ WARNING: No GCP credentials configured")

# -------------------------------------------------
# GCS HELPERS
# -------------------------------------------------

def run_cmd(cmd: List[str], timeout: int = 3600, check: bool = True):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p


def gsutil_ls(pattern: str) -> List[str]:
    p = run_cmd(["gsutil", "ls", pattern], check=False)
    if p.returncode != 0:
        return []
    return [l.strip() for l in p.stdout.splitlines() if l.strip()]


def gsutil_exists(uri: str) -> bool:
    p = subprocess.run(["gsutil", "ls", uri], capture_output=True)
    return p.returncode == 0


def gsutil_cp_to_local(gcs_uri: str, local_path: str):
    run_cmd(["gsutil", "-m", "cp", gcs_uri, local_path])


def gsutil_cp_to_gcs(local_path: str, gcs_uri: str):
    run_cmd(["gsutil", "-m", "cp", local_path, gcs_uri])


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_clip_num(uri: str) -> int:
    m = re.search(r"/Clip-(\d+)/clip-(\d+)\.", uri)
    if not m:
        return 0
    return int(m.group(1))


def guess_mime(uri: str) -> str:
    if uri.lower().endswith(".mov"):
        return "video/quicktime"
    return "video/mp4"


def normalize_tasks(tasks: Any) -> List[Dict[str, str]]:
    if not isinstance(tasks, list):
        return []

    cleaned = []
    for t in tasks:
        if not isinstance(t, dict):
            continue

        start = t.get("start_time") or t.get("start")
        end = t.get("end_time") or t.get("end")
        desc = t.get("description", "")

        if start and end:
            cleaned.append({
                "start_time": str(start).strip(),
                "end_time": str(end).strip(),
                "description": str(desc).strip()
            })

    return cleaned


# -------------------------------------------------
# GEMINI
# -------------------------------------------------

def make_client(project: str, location: str) -> genai.Client:
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
            response_mime_type="application/json"
        )
    )

    tasks = getattr(resp, "parsed", None)

    if tasks is None and hasattr(resp, "text"):
        try:
            tasks = json.loads(resp.text)
        except Exception:
            tasks = []

    return normalize_tasks(tasks)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="global")
    ap.add_argument("--bucket", default="egocentric-main")
    ap.add_argument("--egocentric_prefix", default="Egocentric-Clips")
    ap.add_argument("--tasks_prefix", default="results_egocentric")
    ap.add_argument("--prompt_file", required=True)
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--clip", type=int)
    ap.add_argument("--video_gcs")
    ap.add_argument("--limit", type=int)

    args = ap.parse_args()

    bucket_uri = f"gs://{args.bucket}"
    prompt = load_prompt(args.prompt_file)
    client = make_client(args.project, args.location)

    # -----------------------------
    # DISCOVER VIDEOS (FIXED)
    # -----------------------------

    video_uris: List[str] = []

    if args.video_gcs:
        if not gsutil_exists(args.video_gcs):
            raise RuntimeError(f"Video not found: {args.video_gcs}")
        video_uris = [args.video_gcs]

    elif args.clip is not None:
        uri = f"{bucket_uri}/{args.egocentric_prefix}/Clip-{args.clip}/clip-{args.clip}.mp4"
        if not gsutil_exists(uri):
            raise RuntimeError(f"Video not found: {uri}")
        video_uris = [uri]

    else:
        video_uris = gsutil_ls(
            f"{bucket_uri}/{args.egocentric_prefix}/Clip-*/clip-*.mp4"
        )

    if not video_uris:
        raise RuntimeError("No videos found to process")

    video_uris = sorted(video_uris)
    if args.limit:
        video_uris = video_uris[: args.limit]

    print(f"Found {len(video_uris)} video(s) to process.")

    # -----------------------------
    # PROCESS
    # -----------------------------

    for video_uri in video_uris:
        clip_num = parse_clip_num(video_uri)
        out_uri = f"{bucket_uri}/{args.tasks_prefix}/Clip-{clip_num}_tasks.json"

        print(f"\n▶ Processing: {video_uri}")

        with tempfile.TemporaryDirectory() as td:
            local_video = os.path.join(td, os.path.basename(video_uri))
            gsutil_cp_to_local(video_uri, local_video)

            tasks = generate_tasks(
                client=client,
                model=args.model,
                prompt=prompt,
                video_uri=video_uri,
            )

            local_json = os.path.join(td, "tasks.json")
            with open(local_json, "w") as f:
                json.dump(tasks, f, indent=2)

            gsutil_cp_to_gcs(local_json, out_uri)
            print(f"✓ Uploaded → {out_uri}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
