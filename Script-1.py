#!/usr/bin/env python3
"""
Gemini task segment generator for egocentric clips.

Inputs (GCS):
  gs://<bucket>/Egocentric-Clips/Clip-<N>/clip-<N>.mp4

Output (GCS):
  gs://<bucket>/results_egocentric/Clip-<N>_tasks.json

This JSON is meant to be consumed by the shot cutter script (ffmpeg),
which will create:
  Sessions/Session-<N>/shot_###/*

Requires:
  pip install google-genai
  gcloud auth + permissions to read/write in the bucket
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types


# -----------------------------
# GCS helpers (gsutil-based)
# -----------------------------

def run_cmd(cmd: List[str], timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p


def gsutil_ls(pattern: str) -> List[str]:
    p = run_cmd(["gsutil", "ls", pattern], timeout=3600, check=False)
    if p.returncode != 0:
        return []
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def gsutil_cp_to_local(gcs_uri: str, local_path: str) -> None:
    run_cmd(["gsutil", "-m", "cp", gcs_uri, local_path], timeout=3600)


def gsutil_cp_to_gcs(local_path: str, gcs_uri: str) -> None:
    run_cmd(["gsutil", "-m", "cp", local_path, gcs_uri], timeout=3600)


def gsutil_exists(gcs_uri: str) -> bool:
    p = subprocess.run(["gsutil", "ls", gcs_uri], capture_output=True, text=True)
    return p.returncode == 0 and bool(p.stdout.strip())


# -----------------------------
# Prompt + parsing helpers
# -----------------------------

def load_prompt_local(prompt_file: str) -> str:
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_clip_num_from_uri(uri: str) -> Optional[int]:
    # .../Clip-12/clip-12.mp4
    m = re.search(r"/Clip-(\d+)/clip-(\d+)\.(mp4|m4v|mov)$", uri, flags=re.IGNORECASE)
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    return a if a == b else a


def guess_mime_type(video_uri: str) -> str:
    u = video_uri.lower()
    if u.endswith(".mov"):
        return "video/quicktime"
    return "video/mp4"


def try_extract_json_text_from_response(resp: Any) -> Optional[str]:
    """
    Best-effort: pull model output as raw text that should contain JSON.
    """
    if hasattr(resp, "text") and resp.text:
        return resp.text

    if hasattr(resp, "candidates") and resp.candidates:
        c0 = resp.candidates[0]
        if hasattr(c0, "content") and hasattr(c0.content, "parts") and c0.content.parts:
            p0 = c0.content.parts[0]
            if hasattr(p0, "text") and p0.text:
                return p0.text

    return None


def normalize_tasks(tasks: Any) -> List[Dict[str, Any]]:
    """
    Expected tasks schema is a list of objects with:
      start_time, end_time, description
    We normalize keys and drop invalid items.
    """
    if tasks is None:
        return []

    if not isinstance(tasks, list):
        # If model returned an object, wrap it
        tasks = [tasks]

    cleaned: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue

        # Common key variants
        start = t.get("start_time") or t.get("start") or t.get("startTime")
        end = t.get("end_time") or t.get("end") or t.get("endTime")
        desc = t.get("description") or t.get("task_description") or t.get("desc") or ""

        if start is None or end is None:
            continue

        cleaned.append({
            "start_time": str(start).strip(),
            "end_time": str(end).strip(),
            "description": str(desc).strip(),
        })

    return cleaned


# -----------------------------
# Gemini call
# -----------------------------

def make_genai_client(project: str, location: str) -> genai.Client:
    return genai.Client(vertexai=True, project=project, location=location)


def generate_tasks_for_video(
    client: genai.Client,
    model: str,
    prompt: str,
    video_uri: str,
) -> List[Dict[str, Any]]:
    mime_type = guess_mime_type(video_uri)

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(file_uri=video_uri, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    # First try: parsed
    tasks = getattr(resp, "parsed", None)

    # Fallback: parse text manually
    if tasks is None:
        raw = try_extract_json_text_from_response(resp)
        if raw:
            try:
                tasks = json.loads(raw)
            except json.JSONDecodeError:
                # As a last resort: try to find first JSON array in text
                m = re.search(r"(\[\s*\{.*\}\s*\])", raw, flags=re.DOTALL)
                if m:
                    try:
                        tasks = json.loads(m.group(1))
                    except Exception:
                        tasks = None

    return normalize_tasks(tasks)


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GCP project id used for Vertex AI")
    ap.add_argument("--location", default="global", help="Vertex AI location (global works for availability)")
    ap.add_argument("--bucket", default="egocentric-main", help="Bucket name (no gs://)")
    ap.add_argument("--egocentric_prefix", default="Egocentric-Clips", help="Prefix for main egocentric videos")
    ap.add_argument("--tasks_prefix", default="results_egocentric", help="Prefix to write JSON tasks")
    ap.add_argument("--prompt_file", default="prompts/Egocentric_prompt.txt", help="Local prompt file path on VM")
    ap.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model name")
    ap.add_argument("--clip", type=int, default=None, help="Only process this Clip-<N>")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite tasks JSON if it already exists")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N clips (debug)")
    args = ap.parse_args()

    bucket_uri = f"gs://{args.bucket}"
    prompt = load_prompt_local(args.prompt_file)
    client = make_genai_client(args.project, args.location)

    # Discover video uris
    if args.clip is not None:
        video_uris = [f"{bucket_uri}/{args.egocentric_prefix}/Clip-{args.clip}/clip-{args.clip}.mp4"]
        if not gsutil_exists(video_uris[0]):
            raise RuntimeError(f"Video not found: {video_uris[0]}")
    else:
        video_uris = gsutil_ls(f"{bucket_uri}/{args.egocentric_prefix}/Clip-*/clip-*.mp4")
        if not video_uris:
            raise RuntimeError(f"No videos found at {bucket_uri}/{args.egocentric_prefix}/Clip-*/clip-*.mp4")

    video_uris = sorted(video_uris)
    if args.limit is not None:
        video_uris = video_uris[: args.limit]

    print(f"Found {len(video_uris)} video(s) to process.")

    for video_uri in video_uris:
        clip_num = parse_clip_num_from_uri(video_uri)
        if clip_num is None:
            print(f"[WARN] Could not parse clip number from: {video_uri}")
            continue

        out_tasks_uri = f"{bucket_uri}/{args.tasks_prefix}/Clip-{clip_num}_tasks.json"

        if (not args.overwrite) and gsutil_exists(out_tasks_uri):
            print(f"[SKIP] tasks already exist: {out_tasks_uri}")
            continue

        print(f"\n=== Clip-{clip_num} ===")
        print(f"Video: {video_uri}")
        print(f"Out:   {out_tasks_uri}")

        try:
            tasks = generate_tasks_for_video(
                client=client,
                model=args.model,
                prompt=prompt,
                video_uri=video_uri,
            )
        except Exception as e:
            print(f"[ERROR] Gemini failed for Clip-{clip_num}: {e}")
            continue

        print(f"Tasks returned: {len(tasks)}")

        # Write locally then upload to GCS
        with tempfile.TemporaryDirectory() as td:
            local_json = os.path.join(td, f"Clip-{clip_num}_tasks.json")
            with open(local_json, "w", encoding="utf-8") as f:
                json.dump(tasks, f, indent=2)

            # Upload
            try:
                gsutil_cp_to_gcs(local_json, out_tasks_uri)
                print(f"[OK] Uploaded tasks -> {out_tasks_uri}")
            except Exception as e:
                print(f"[ERROR] Upload failed for Clip-{clip_num}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
