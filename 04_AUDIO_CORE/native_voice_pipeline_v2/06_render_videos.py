#!/usr/bin/env python3
"""Render EP1-3 videos using avatar.png with Ken-Burns + audio overlay."""
import subprocess, os
from pathlib import Path

AVATAR = Path("/home/user/workspace/sisi-lola-project/sisi_lola_website/assets/sisi-lola-avatar.png")
AUDIO_DIR = Path("/home/user/workspace/voice-clone/output_v2")
OUT_DIR = Path("/home/user/workspace/voice-clone/video_v2")
OUT_DIR.mkdir(exist_ok=True)

eps = ["ep1-hustle", "ep2-food", "ep3-language"]

for ep in eps:
    audio = AUDIO_DIR / f"{ep}.mp3"
    out = OUT_DIR / f"{ep}-video.mp4"
    # Probe duration
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(audio)
    ]).strip())
    print(f"{ep}: audio dur = {dur:.1f}s")
    
    # Ken-Burns: slow zoom-in over duration
    # 1080x1080 source -> 1920x1080 output, gentle zoompan 1.0->1.15
    fps = 25
    nframes = int(dur * fps)
    # zoompan zoom expression: 1 + 0.15 * t/totaldur
    # use d=1 for per-frame zoom
    vf = (
        f"scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,"
        f"zoompan=z='min(zoom+0.0006,1.18)':d={nframes}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={fps},"
        f"format=yuv420p"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(AVATAR),
        "-i", str(audio),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(out)
    ]
    print(f"  rendering {out.name}...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[-500:]}")
    else:
        size_mb = out.stat().st_size / 1e6
        print(f"  OK: {out} ({size_mb:.1f} MB)")

print("\nAll videos rendered.")
