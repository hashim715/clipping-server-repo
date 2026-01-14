# #!/usr/bin/env python3
# """
# Gemini task segment generator + FFmpeg clipper for egocentric clips.

# Inputs (GCS):
#   gs://<bucket>/Egocentric-Clips/Clip-<N>/clip-<N>.mp4

# Output:
#   1) Task JSON: gs://<bucket>/results_egocentric/Clip-<N>_tasks.json
#   2) Clips: gs://<bucket>/Sessions/Session-<N>/clip_###.mp4
# """

# import argparse
# import json
# import os
# import re
# import subprocess
# import tempfile
# from typing import Any, Dict, List, Optional

# from google import genai
# from google.genai import types

# # -----------------------------
# # GCS helpers
# # -----------------------------
# def run_cmd(cmd: List[str], timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
#     p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
#     if check and p.returncode != 0:
#         raise RuntimeError(
#             f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
#         )
#     return p

# def gsutil_ls(pattern: str) -> List[str]:
#     p = run_cmd(["gsutil", "ls", pattern], timeout=3600, check=False)
#     if p.returncode != 0:
#         return []
#     return [line.strip() for line in p.stdout.splitlines() if line.strip()]

# def gsutil_cp_to_local(gcs_uri: str, local_path: str) -> None:
#     run_cmd(["gsutil", "-m", "cp", gcs_uri, local_path], timeout=3600)

# def gsutil_cp_to_gcs(local_path: str, gcs_uri: str) -> None:
#     run_cmd(["gsutil", "-m", "cp", local_path, gcs_uri], timeout=3600)

# def gsutil_exists(gcs_uri: str) -> bool:
#     p = subprocess.run(["gsutil", "ls", gcs_uri], capture_output=True, text=True)
#     return p.returncode == 0 and bool(p.stdout.strip())

# # -----------------------------
# # Utilities
# # -----------------------------
# def parse_clip_num_from_uri(uri: str) -> Optional[int]:
#     m = re.search(r"/Clip-(\d+)/clip-(\d+)\.(mp4|m4v|mov)$", uri, flags=re.IGNORECASE)
#     if not m:
#         return None
#     a, b = int(m.group(1)), int(m.group(2))
#     return a if a == b else a

# def guess_mime_type(video_uri: str) -> str:
#     u = video_uri.lower()
#     if u.endswith(".mov"):
#         return "video/quicktime"
#     return "video/mp4"

# def load_prompt_local(prompt_file: str) -> str:
#     if not os.path.exists(prompt_file):
#         raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
#     with open(prompt_file, "r", encoding="utf-8") as f:
#         return f.read().strip()

# def normalize_tasks(tasks: Any) -> List[Dict[str, Any]]:
#     if tasks is None:
#         return []

#     if not isinstance(tasks, list):
#         tasks = [tasks]

#     cleaned: List[Dict[str, Any]] = []
#     for t in tasks:
#         if not isinstance(t, dict):
#             continue
#         start = t.get("start_time") or t.get("start") or t.get("startTime")
#         end = t.get("end_time") or t.get("end") or t.get("endTime")
#         desc = t.get("description") or t.get("task_description") or t.get("desc") or ""
#         if start is None or end is None:
#             continue
#         cleaned.append({
#             "start_time": str(start).strip(),
#             "end_time": str(end).strip(),
#             "description": str(desc).strip(),
#         })
#     return cleaned

# def try_extract_json_text_from_response(resp: Any) -> Optional[str]:
#     if hasattr(resp, "text") and resp.text:
#         return resp.text
#     if hasattr(resp, "candidates") and resp.candidates:
#         c0 = resp.candidates[0]
#         if hasattr(c0, "content") and hasattr(c0.content, "parts") and c0.content.parts:
#             p0 = c0.content.parts[0]
#             if hasattr(p0, "text") and p0.text:
#                 return p0.text
#     return None

# # -----------------------------
# # Gemini
# # -----------------------------
# def make_genai_client(project: str, location: str) -> genai.Client:
#     return genai.Client(vertexai=True, project=project, location=location)

# def generate_tasks_for_video(
#     client: genai.Client,
#     model: str,
#     prompt: str,
#     video_uri: str,
# ) -> List[Dict[str, Any]]:
#     mime_type = guess_mime_type(video_uri)
#     resp = client.models.generate_content(
#         model=model,
#         contents=[
#             types.Part.from_uri(file_uri=video_uri, mime_type=mime_type),
#             prompt,
#         ],
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#         ),
#     )
#     tasks = getattr(resp, "parsed", None)
#     if tasks is None:
#         raw = try_extract_json_text_from_response(resp)
#         if raw:
#             try:
#                 tasks = json.loads(raw)
#             except Exception:
#                 m = re.search(r"(\[\s*\{.*\}\s*\])", raw, flags=re.DOTALL)
#                 if m:
#                     try:
#                         tasks = json.loads(m.group(1))
#                     except Exception:
#                         tasks = None
#     return normalize_tasks(tasks)

# # -----------------------------
# # FFmpeg clipper
# # -----------------------------
# def ffmpeg_clip(local_video: str, start: str, end: str, output_path: str) -> None:
#     """
#     Clips video from start to end timestamps using FFmpeg
#     """
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", local_video,
#         "-ss", start,
#         "-to", end,
#         "-c", "copy",
#         output_path
#     ]
#     run_cmd(cmd)

# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--project", required=True)
#     ap.add_argument("--location", default="global")
#     ap.add_argument("--bucket", default="egocentric-main")
#     ap.add_argument("--egocentric_prefix", default="Egocentric-Clips")
#     ap.add_argument("--tasks_prefix", default="results_egocentric")
#     ap.add_argument("--prompt_file", default="prompts/Egocentric_prompt.txt")
#     ap.add_argument("--model", default="gemini-3-flash-preview")
#     ap.add_argument("--clip", type=int, default=None)
#     ap.add_argument("--overwrite", action="store_true")
#     ap.add_argument("--limit", type=int, default=None)
#     args = ap.parse_args()

#     bucket_uri = f"gs://{args.bucket}"
#     prompt = load_prompt_local(args.prompt_file)
#     client = make_genai_client(args.project, args.location)

#     if args.clip is not None:
#         video_uris = [f"{bucket_uri}/{args.egocentric_prefix}/Clip-{args.clip}/clip-{args.clip}.mp4"]
#         if not gsutil_exists(video_uris[0]):
#             raise RuntimeError(f"Video not found: {video_uris[0]}")
#     else:
#         video_uris = gsutil_ls(f"{bucket_uri}/{args.egocentric_prefix}/Clip-*/clip-*.mp4")
#         if not video_uris:
#             raise RuntimeError(f"No videos found at {bucket_uri}/{args.egocentric_prefix}/Clip-*/clip-*.mp4")

#     video_uris = sorted(video_uris)
#     if args.limit is not None:
#         video_uris = video_uris[: args.limit]

#     print(f"Found {len(video_uris)} video(s) to process.")

#     for video_uri in video_uris:
#         clip_num = parse_clip_num_from_uri(video_uri)
#         if clip_num is None:
#             print(f"[WARN] Could not parse clip number from: {video_uri}")
#             continue

#         out_tasks_uri = f"{bucket_uri}/{args.tasks_prefix}/Clip-{clip_num}_tasks.json"
#         if (not args.overwrite) and gsutil_exists(out_tasks_uri):
#             print(f"[SKIP] tasks already exist: {out_tasks_uri}")
#             continue

#         print(f"\n=== Clip-{clip_num} ===\nVideo: {video_uri}\nOut JSON: {out_tasks_uri}")

#         try:
#             tasks = generate_tasks_for_video(
#                 client=client,
#                 model=args.model,
#                 prompt=prompt,
#                 video_uri=video_uri,
#             )
#         except Exception as e:
#             print(f"[ERROR] Gemini failed: {e}")
#             continue

#         print(f"Tasks returned: {len(tasks)}")

#         with tempfile.TemporaryDirectory() as td:
#             local_video = os.path.join(td, f"clip-{clip_num}.mp4")
#             gsutil_cp_to_local(video_uri, local_video)

#             # Save JSON locally
#             local_json = os.path.join(td, f"Clip-{clip_num}_tasks.json")
#             with open(local_json, "w", encoding="utf-8") as f:
#                 json.dump(tasks, f, indent=2)
#             gsutil_cp_to_gcs(local_json, out_tasks_uri)

#             # Clip videos using FFmpeg
#             for idx, task in enumerate(tasks, 1):
#                 start, end = task["start_time"], task["end_time"]
#                 clip_name = f"clip_{idx:03d}.mp4"
#                 local_clip = os.path.join(td, clip_name)
#                 ffmpeg_clip(local_video, start, end, local_clip)

#                 # Upload each clip to GCS
#                 gcs_clip_path = f"{bucket_uri}/Sessions/Session-{clip_num}/{clip_name}"
#                 gsutil_cp_to_gcs(local_clip, gcs_clip_path)
#                 print(f"[OK] Uploaded clip {clip_name} -> {gcs_clip_path}")

#     print("\nDone.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
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
from typing import List, Dict

load_dotenv()

if os.getenv("GCP_KEY_FILE") and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    gcp_key_path = Path(os.getenv("GCP_KEY_FILE")).resolve()
    if not gcp_key_path.exists():
        raise FileNotFoundError(f"GCP key file not found: {gcp_key_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(gcp_key_path)
    print(f"✓ GCP credentials loaded from: {gcp_key_path}")
elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    print(f"✓ Using existing GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
else:
    print("⚠ Warning: No GCP credentials configured")

def run_cmd(cmd: list, timeout: int = None, check: bool = True):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p

def gsutil_cp_to_local(gcs_uri: str, local_path: str):
    print(f"Downloading {gcs_uri} -> {local_path}")
    run_cmd(["gsutil", "-m", "cp", gcs_uri, local_path])

def gsutil_cp_to_gcs(local_path: str, gcs_uri: str):
    print(f"Uploading {local_path} -> {gcs_uri}")
    run_cmd(["gsutil", "-m", "cp", local_path, gcs_uri])

def ffmpeg_clip(local_video: str, start: str, end: str, output_path: str):
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
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_gcs", required=True, help="GCS URL of the video to clip")
    parser.add_argument("--json_gcs", required=True, help="GCS URL of the JSON defining segments")
    parser.add_argument("--clips_gcs_prefix", required=True, help="GCS path prefix to upload clips (e.g., gs://bucket/Sessions/Session-1)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        local_video = os.path.join(td, "video.mp4")
        gsutil_cp_to_local(args.video_gcs, local_video)

        local_json = os.path.join(td, "segments.json")
        gsutil_cp_to_local(args.json_gcs, local_json)

        with open(local_json, "r", encoding="utf-8") as f:
            segments = json.load(f)

        for idx, seg in enumerate(segments, start=1):
            start = seg["start_time"]
            end = seg["end_time"]
            clip_name = f"shot_{idx:03d}.mp4"
            local_clip = os.path.join(td, clip_name)
            ffmpeg_clip(local_video, start, end, local_clip)

            gcs_clip_path = f"{args.clips_gcs_prefix}/{clip_name}"
            gsutil_cp_to_gcs(local_clip, gcs_clip_path)
            print(f"[OK] Uploaded {clip_name} -> {gcs_clip_path}")

    print("\nAll clips processed and uploaded.")

if __name__ == "__main__":
    main()
