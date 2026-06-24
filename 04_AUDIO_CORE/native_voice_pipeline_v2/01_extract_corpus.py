#!/usr/bin/env python3
"""
Download one parquet shard at a time from EYEDOL/naija-voices-yoruba split, 
filter female + 30-over (warm big-sis vibe matching Sisi Lola), save as WAV.
Bypasses torchcodec entirely — uses pyarrow + soundfile directly.
"""
import os, io, sys
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np
from huggingface_hub import hf_hub_download

OUT_DIR = Path("/home/user/workspace/voice-clone/final_corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_FILE = OUT_DIR / "transcripts.tsv"
TARGET_MINUTES = 35.0
MIN_S = 4.0
MAX_S = 18.0
MAX_PER_SPEAKER = 25

# Try multiple shards across splits to maximize speaker diversity
SHARDS = []
for split_idx in [5, 4, 0, 1, 3]:
    repo = f"EYEDOL/naija-voices-yoruba-split_0-{split_idx}"
    for n in range(10):
        SHARDS.append((repo, f"data/train-{n:05d}-of-00010.parquet"))

total_s = 0.0
saved = 0
seen_speakers = {}

tf = open(TRANSCRIPT_FILE, "w", encoding="utf-8")
tf.write("filename\tspeaker_id\tage_range\tduration\ttext\n")

for repo, fname in SHARDS:
    if total_s >= TARGET_MINUTES * 60:
        break
    
    print(f"\n=== {repo} :: {fname} ===", flush=True)
    try:
        local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
    except Exception as e:
        print(f"  ! download failed: {e}", flush=True)
        continue
    
    try:
        table = pq.read_table(local)
    except Exception as e:
        print(f"  ! read failed: {e}", flush=True)
        continue
    
    print(f"  rows={table.num_rows}, cols={table.column_names}", flush=True)
    
    rows = table.to_pylist()
    shard_saved = 0
    
    for idx, row in enumerate(rows):
        if total_s >= TARGET_MINUTES * 60:
            break
        if row.get("gender") != "female":
            continue
        if row.get("age_range") != "30-over":
            continue
        
        # audio column is {bytes: ..., path: ...}
        audio = row.get("audio")
        if not audio:
            continue
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            continue
        
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            continue
        
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        dur = len(arr) / sr
        if dur < MIN_S or dur > MAX_S:
            continue
        
        peak = float(np.max(np.abs(arr)))
        rms = float(np.sqrt(np.mean(arr**2)))
        if peak < 0.05 or rms < 0.01 or peak > 0.99:
            continue
        
        spk = row.get("speaker_id", "unk")
        seen_speakers[spk] = seen_speakers.get(spk, 0) + 1
        if seen_speakers[spk] > MAX_PER_SPEAKER:
            continue
        
        out_name = f"{spk}_{saved:05d}.wav"
        sf.write(OUT_DIR / out_name, arr, sr, subtype="PCM_16")
        text = (row.get("text") or "").replace("\t", " ").replace("\n", " ").strip()
        tf.write(f"{out_name}\t{spk}\t{row.get('age_range')}\t{dur:.2f}\t{text}\n")
        tf.flush()
        
        saved += 1
        shard_saved += 1
        total_s += dur
    
    # Delete the parquet shard to save disk
    try:
        os.remove(local)
    except Exception:
        pass
    
    print(f"  shard yielded {shard_saved} female-30+ clips; total {saved}, {total_s/60:.1f} min, {len(seen_speakers)} speakers", flush=True)

tf.close()
print(f"\n=== DONE === {saved} clips, {total_s/60:.1f} min, {len(seen_speakers)} unique speakers")
print("\nTop speakers:")
for spk, n in sorted(seen_speakers.items(), key=lambda x: -x[1])[:15]:
    print(f"  {spk}: {n} clips")
