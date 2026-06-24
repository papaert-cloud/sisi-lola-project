"""Re-render all 3 Sisi Lola episodes using new native-voice audio + existing keyframes.

Strategy: Ken-Burns crossfade between keyframe-1 and keyframe-2 of each episode,
synced to the full duration of the new audio. Adds Sisi Lola brand intro/outro
fade. Output: 1920x1080 H.264 MP4 at 30fps.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/user/workspace")
KEY = ROOT / "sisi-lola-video-keyframes"
AUDIO = ROOT / "voice-clone" / "output"
OUT = ROOT / "sisi-lola-site" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

EPISODES = [
    ("ep1", "ep1-hustle.wav",   "ep1-video.mp4"),
    ("ep2", "ep2-food.wav",     "ep2-video.mp4"),
    ("ep3", "ep3-language.wav", "ep3-video.mp4"),
]

def get_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


for ep, audio_name, out_name in EPISODES:
    audio_path = AUDIO / audio_name
    kf1 = KEY / ep / "keyframe-1.png"
    kf2 = KEY / ep / "keyframe-2.png"
    out_path = OUT / out_name

    duration = get_duration(audio_path)
    half = duration / 2

    print(f"\n=== {ep} ===  audio: {audio_name}  duration: {duration:.2f}s")
    print(f"  keyframes: {kf1.name}, {kf2.name}")
    print(f"  -> {out_path.name}")

    # Build a video with Ken-Burns zoom on each keyframe, crossfading mid-episode.
    # Using zoompan to slowly zoom in on each frame, then xfade between them.
    fps = 30
    half_frames = int(half * fps)

    # Ken-Burns: slow zoom in to 1.15x over half_frames, then crossfade
    filter_complex = (
        f"[0:v]scale=2400:1350,zoompan=z='min(zoom+0.0005,1.15)':d={half_frames}:s=1920x1080:fps={fps},setsar=1[v1];"
        f"[1:v]scale=2400:1350,zoompan=z='min(zoom+0.0005,1.15)':d={half_frames}:s=1920x1080:fps={fps},setsar=1[v2];"
        f"[v1][v2]xfade=transition=fade:duration=1.5:offset={half-0.75}[vout];"
        # Tiny fade-in/out at start/end for polish
        f"[vout]fade=t=in:st=0:d=1,fade=t=out:st={duration-1}:d=1[final]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{half}", "-i", str(kf1),
        "-loop", "1", "-t", f"{half}", "-i", str(kf2),
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[final]", "-map", "2:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-r", str(fps),
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-1500:]}")
        sys.exit(1)
    final_size = out_path.stat().st_size / 1024 / 1024
    print(f"  DONE: {final_size:.1f} MB")

print("\nAll 3 videos re-rendered.")
