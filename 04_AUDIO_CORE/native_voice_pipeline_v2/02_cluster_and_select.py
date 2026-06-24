#!/usr/bin/env python3
"""
Embed all corpus clips with Resemblyzer (speaker encoder),
cluster to find the largest acoustically-coherent group of female voices,
output the curated subset for ElevenLabs PVC upload.
"""
import os
from pathlib import Path
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import csv

CORPUS = Path("/home/user/workspace/voice-clone/final_corpus")
SELECTED = Path("/home/user/workspace/voice-clone/pvc_selected")
SELECTED.mkdir(exist_ok=True)

# Read transcript
clips = []
with open(CORPUS / "transcripts.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        clips.append(row)

print(f"Loaded {len(clips)} clips")

# Compute embeddings for all clips (CPU)
print("Loading Resemblyzer (CPU)...")
encoder = VoiceEncoder("cpu")
print("Embedding clips...")
embeds = []
keep = []
for i, c in enumerate(clips):
    if i % 50 == 0:
        print(f"  {i}/{len(clips)}", flush=True)
    try:
        wav = preprocess_wav(CORPUS / c["filename"])
        e = encoder.embed_utterance(wav)
        embeds.append(e)
        keep.append(c)
    except Exception as ex:
        print(f"  skip {c['filename']}: {ex}")
        continue
embeds = np.array(embeds)
print(f"Embeddings: {embeds.shape}")

# Cluster
print("\nClustering with cosine distance...")
# Try several cluster counts, pick by silhouette
best_n, best_score, best_labels = None, -1, None
for n in [3, 4, 5, 6, 8, 10]:
    clst = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    labels = clst.fit_predict(embeds)
    try:
        score = silhouette_score(embeds, labels, metric="cosine")
    except Exception:
        score = -1
    print(f"  n={n}: silhouette={score:.3f}")
    if score > best_score:
        best_score = score
        best_labels = labels
        best_n = n

print(f"\nBest n_clusters={best_n}, silhouette={best_score:.3f}")

# Find largest cluster (by total duration)
from collections import defaultdict
clust_dur = defaultdict(float)
clust_clips = defaultdict(list)
for c, lab in zip(keep, best_labels):
    clust_dur[lab] += float(c["duration"])
    clust_clips[lab].append(c)

ranked = sorted(clust_dur.items(), key=lambda x: -x[1])
print(f"\nClusters by total duration:")
for lab, d in ranked:
    n = len(clust_clips[lab])
    spk = set(c["speaker_id"] for c in clust_clips[lab])
    print(f"  cluster {lab}: {n} clips, {d/60:.1f} min, {len(spk)} unique speakers")

# Pick the largest cluster
top_lab = ranked[0][0]
chosen = clust_clips[top_lab]
print(f"\n=== Selected cluster {top_lab}: {len(chosen)} clips, {sum(float(c['duration']) for c in chosen)/60:.1f} min ===")

# Copy to selected dir + write new transcript
out_tsv = SELECTED / "transcripts.tsv"
with open(out_tsv, "w") as f:
    f.write("filename\tspeaker_id\tduration\ttext\n")
    for c in chosen:
        src = CORPUS / c["filename"]
        dst = SELECTED / c["filename"]
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
        f.write(f"{c['filename']}\t{c['speaker_id']}\t{c['duration']}\t{c['text']}\n")

print(f"\nSaved to {SELECTED}")
