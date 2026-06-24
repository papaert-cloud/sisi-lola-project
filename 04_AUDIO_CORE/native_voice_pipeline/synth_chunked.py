"""Synthesize Sisi Lola episodes using XTTS-v2, chunk-by-chunk for fast CPU progress.

We split each script into short sentence groups (~120 chars each) so each XTTS
call processes a manageable token count. Results are concatenated with ffmpeg.

CPU XTTS-v2: ~3-5x realtime on 8 vCPU. ~80s script -> ~5min synth.
"""
import os
import re
import sys
import time
import subprocess
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

# Set torch threads
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

from TTS.api import TTS

REFERENCE = "/home/user/workspace/voice-clone/references/sisi_lola_native_yoruba_reference.wav"
SCRIPTS_DIR = Path("/home/user/workspace/sisi-lola-site/scripts")
OUTPUT_DIR = Path("/home/user/workspace/voice-clone/output")
CHUNKS_DIR = OUTPUT_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = [
    ("ep1-no-give-up.txt", "ep1-hustle"),
    ("ep2-chop-life.txt",  "ep2-food"),
    ("ep3-ede-wa.txt",     "ep3-language"),
]


def split_into_chunks(text: str, max_chars: int = 140):
    """Split text into chunks at sentence boundaries, ~max_chars each."""
    # Sentences end with . ! ? but not at . in middle of word
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n\n", " ").replace("\n", " "))
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            # If single sentence is longer than max_chars, hard-split at commas
            if len(s) > max_chars:
                parts = re.split(r'(?<=,)\s+', s)
                cur2 = ""
                for p in parts:
                    if len(cur2) + len(p) + 1 <= max_chars:
                        cur2 = (cur2 + " " + p).strip()
                    else:
                        if cur2:
                            chunks.append(cur2)
                        cur2 = p
                if cur2:
                    chunks.append(cur2)
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)
    return chunks


print(f"Threads: {torch.get_num_threads()}")
print("Loading XTTS-v2...")
t0 = time.time()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cpu")
print(f"Model loaded in {time.time()-t0:.1f}s\n", flush=True)


def synth_episode(script_name: str, out_basename: str):
    script_path = SCRIPTS_DIR / script_name
    text = script_path.read_text(encoding="utf-8").strip()
    chunks = split_into_chunks(text, max_chars=140)
    print(f"=== {out_basename} ===  ({len(chunks)} chunks)", flush=True)

    chunk_paths = []
    ep_dir = CHUNKS_DIR / out_basename
    ep_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_path = ep_dir / f"chunk_{i:02d}.wav"
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  [{i+1:02d}/{len(chunks)}] cached {out_path.name}", flush=True)
            chunk_paths.append(str(out_path))
            continue
        t0 = time.time()
        try:
            tts.tts_to_file(
                text=chunk,
                speaker_wav=REFERENCE,
                language="en",
                file_path=str(out_path),
                split_sentences=False,
            )
            elapsed = time.time() - t0
            print(f"  [{i+1:02d}/{len(chunks)}] {len(chunk):3d}c -> {out_path.name} in {elapsed:.1f}s  | '{chunk[:60]}...'", flush=True)
            chunk_paths.append(str(out_path))
        except Exception as e:
            print(f"  [{i+1:02d}/{len(chunks)}] FAILED: {e}", flush=True)

    # Concatenate via ffmpeg
    concat_list = ep_dir / "concat.txt"
    with concat_list.open("w") as f:
        for p in chunk_paths:
            f.write(f"file '{p}'\n")

    final_path = OUTPUT_DIR / f"{out_basename}.wav"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-filter_complex", "[0:a]aresample=22050,aformat=channel_layouts=mono,loudnorm=I=-18:TP=-2[out]",
        "-map", "[out]", "-c:a", "pcm_s16le",
        str(final_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg concat error: {result.stderr[-500:]}")
    else:
        print(f"  CONCATENATED -> {final_path.name}", flush=True)

    # Also produce MP3 for web delivery
    mp3_path = OUTPUT_DIR / f"{out_basename}.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", str(final_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3_path)],
                   capture_output=True)
    print(f"  MP3 -> {mp3_path.name}\n", flush=True)


for script_name, out_basename in EPISODES:
    synth_episode(script_name, out_basename)

print("All episodes done.")
