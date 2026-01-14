import subprocess
import os
import sys
from pathlib import Path


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def split_video(video_path, segment_duration=900):
    """
    Split video into segments of specified duration (default 15 minutes = 900 seconds)
    
    Args:
        video_path: Path to input video
        segment_duration: Duration of each segment in seconds (default: 900 = 15 minutes)
    """
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Get video duration
    try:
        total_duration = get_video_duration(video_path)
        print(f"Video duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return
    
    # Setup output directory and filenames
    video_path_obj = Path(video_path)
    output_dir = video_path_obj.parent / "egocentric-clips"
    output_dir.mkdir(exist_ok=True)
    
    output_pattern = str(output_dir / f"clip-%d{video_path_obj.suffix}")
    
    print(f"\nSplitting video into {segment_duration} second ({segment_duration/60} minute) segments...")
    print(f"Output directory: {output_dir}")
    
    # FFmpeg command to split video
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-c', 'copy',  # Copy codec without re-encoding (fast)
        '-map', '0',   # Map all streams
        '-segment_time', str(segment_duration),
        '-f', 'segment',
        '-reset_timestamps', '1',
        output_pattern
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Video split successfully!")
        print(f"Output files saved to: {output_dir}")
        
        # List created files
        clips = sorted(output_dir.glob(f"clip-*{video_path_obj.suffix}"))
        print(f"\nCreated {len(clips)} clips:")
        for clip in clips:
            size_mb = clip.stat().st_size / (1024 * 1024)
            print(f"  - {clip.name} ({size_mb:.2f} MB)")
            
    except subprocess.CalledProcessError as e:
        print(f"Error splitting video: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_splitter.py <video_path> [duration_in_minutes]")
        print("Example: python video_splitter.py /path/to/video.mp4 15")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Optional: custom duration in minutes
    if len(sys.argv) >= 3:
        duration_minutes = float(sys.argv[2])
        segment_duration = int(duration_minutes * 60)
    else:
        segment_duration = 900  # Default 15 minutes
    
    split_video(video_path, segment_duration)