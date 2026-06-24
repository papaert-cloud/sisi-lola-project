#!/usr/bin/env python3
"""
Strategy: Find speakers in our corpus who acoustically match the Sisi Lola archetype.

Use Resemblyzer to embed:
- The existing reference clip (FLEURS-derived, our first cut)
- All corpus clips
Then rank corpus clips by cosine similarity to the reference.
Output: top-K most-similar clips (target 5-10 min) — these become the IVC corpus.
"""
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav

REF = Path("/home/user/workspace/sisi-lola-project/04_AUDIO_CORE/native_voice_pipeline/references/sisi_lola_native_yoruba_reference.wav")
CORPUS = Path("/home/user/workspace/voice-clone/final_corpus")
OUT = Path("/home/user/workspace/voice-clone/ivc_topk")
OUT.mkdir(exist_ok=True)

TARGET_MIN = 8.0  # IVC sweet spot: 3-10 min; we'll aim for 8

print("Loading encoder...")
enc = VoiceEncoder("cpu")

print(f"Embedding reference: {REF.name}")
ref_wav = preprocess_wav(REF)
ref_emb = enc.embed_utterance(ref_wav)

# Embed all corpus clips
print("Embedding corpus...")
clips = []
with open(CORPUS / "transcripts.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        clips.append(row)

embeds = []
keep = []
for i, c in enumerate(clips):
    if i % 50 == 0:
        print(f"  {i}/{len(clips)}", flush=True)
    try:
        wav = preprocess_wav(CORPUS / c["filename"])
        e = enc.embed_utterance(wav)
        embeds.append(e)
        keep.append(c)
    except Exception:
        continue
embeds = np.array(embeds)

# Cosine similarity to reference
ref_emb_n = ref_emb / np.linalg.norm(ref_emb)
embeds_n = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
sims = embeds_n @ ref_emb_n

# Sort descending
order = np.argsort(-sims)

# Collect highest-similarity clips until we hit TARGET_MIN
chosen = []
total_s = 0.0
spk_count = {}
for idx in order:
    c = keep[idx]
    d = float(c["duration"])
    spk = c["speaker_id"]
    # Encourage diversity but allow up to 5 per speaker to keep timbre consistent
    if spk_count.get(spk, 0) >= 8:
        continue
    chosen.append((idx, c, float(sims[idx])))
    total_s += d
    spk_count[spk] = spk_count.get(spk, 0) + 1
    if total_s >= TARGET_MIN * 60:
        break

print(f"\nSelected {len(chosen)} clips, {total_s/60:.1f} min, {len(spk_count)} unique speakers")
print("Top-25 most-similar clips:")
for idx, c, s in chosen[:25]:
    print(f"  sim={s:.3f} spk={c['speaker_id']} dur={c['duration']}s :: {c['text'][:60]}")

# Copy clips, build transcript
import shutil
with open(OUT / "transcripts.tsv", "w") as f:
    f.write("filename\tspeaker_id\tsimilarity\tduration\ttext\n")
    for idx, c, s in chosen:
        src = CORPUS / c["filename"]
        dst = OUT / c["filename"]
        if not dst.exists():
            shutil.copy2(src, dst)
        f.write(f"{c['filename']}\t{c['speaker_id']}\t{s:.4f}\t{c['duration']}\t{c['text']}\n")

print(f"\nSaved {len(chosen)} clips to {OUT}")

# Now also produce a SINGLE concatenated reference (with 0.3s gaps), loudness-normalized
print("\nBuilding concatenated IVC reference clip...")
import subprocess
filelist = OUT / "concat_list.txt"
with open(filelist, "w") as f:
    silence = OUT / "_silence.wav"
    # Generate a brief silence
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", "0.3", str(silence)], capture_output=True, check=True)
    for i, (_, c, _) in enumerate(chosen):
        f.write(f"file '{(OUT/c['filename']).as_posix()}'\n")
        if i < len(chosen) - 1:
            f.write(f"file '{silence.as_posix()}'\n")

big_ref = OUT / "sisi_lola_archetype_reference.wav"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(filelist),
    "-af", "loudnorm=I=-18:TP=-2:LRA=11,aresample=24000",
    "-ac", "1", "-ar", "24000",
    str(big_ref)
], capture_output=True, check=True)
print(f"Built {big_ref} ({big_ref.stat().st_size//1024} KB)")
