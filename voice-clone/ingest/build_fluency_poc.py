"""Build PoC fluency-progression chart V2 -> V3 -> V4 toward native (=100).
Outputs PNG at /home/user/workspace/sisi_lola_fluency_poc.png.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MANI = Path("/home/user/workspace/voice-clone/corpus_v5/manifests/fluency_scores_v2.jsonl")
records = [json.loads(l) for l in MANI.read_text().splitlines() if l.strip()]
by = {}
for r in records:
    by[r["label"]] = r

# Per-version aggregate
def avg(prefix):
    eps = [by[f"{prefix}_ep{i}"] for i in (1,2,3) if f"{prefix}_ep{i}" in by]
    return [e["score"] for e in eps]

native = by["W8MQ2_GROUND_TRUTH"]["score"]
versions = {
    "V2\nYorunglish\n(ElevenLabs)": avg("V2_Yorunglish"),
    "V3\nPure Yoruba\n(ElevenLabs)": avg("V3_ElevenLabs"),
    "V4\nF5-TTS\nNative-trained": avg("V4_F5TTS"),
}

fig = plt.figure(figsize=(14, 6.5))
fig.patch.set_facecolor("#FAFAF7")
ax1 = fig.add_subplot(1, 2, 1)

# ===== LEFT: bar chart of mean scores =====
labels = list(versions.keys()) + ["W8MQ2\nGround Truth\n(Native Speaker)"]
means  = [np.mean(versions[k]) for k in versions] + [native]
mins   = [min(versions[k])     for k in versions] + [native]
maxs   = [max(versions[k])     for k in versions] + [native]

colors = ["#E5894B", "#D8A14C", "#7A9F6E", "#3F8A4D"]
x = np.arange(len(labels))
bars = ax1.bar(x, means, color=colors, edgecolor="#222", linewidth=1.2, width=0.62, zorder=3)
# Min-max whiskers
for i, (lo, hi) in enumerate(zip(mins, maxs)):
    ax1.plot([i, i], [lo, hi], color="#222", lw=1.6, zorder=4)
    ax1.plot([i-0.08, i+0.08], [lo, lo], color="#222", lw=1.6, zorder=4)
    ax1.plot([i-0.08, i+0.08], [hi, hi], color="#222", lw=1.6, zorder=4)

# Native target line
ax1.axhline(100, color="#3F8A4D", ls="--", lw=1.5, alpha=0.7, zorder=2)
ax1.text(len(labels)-0.55, 100.6, "100 = native baseline", color="#3F8A4D",
         fontsize=9, fontweight="bold", ha="right")
ax1.axhline(92,  color="#7A9F6E", ls=":",  lw=1.2, alpha=0.6, zorder=2)
ax1.text(0, 92.5, "≥92 native-indistinguishable", color="#5A7A55", fontsize=8.5, alpha=0.8)
ax1.axhline(85,  color="#C8A04A", ls=":",  lw=1.0, alpha=0.5, zorder=2)
ax1.text(0, 85.5, "≥85 production-ready", color="#8A7234", fontsize=8.5, alpha=0.7)

for bar, m in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, m + 0.6, f"{m:.1f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1A1A1A")

ax1.set_ylim(70, 104)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("Fluency Score (0–100, 100 = native)", fontsize=11, fontweight="bold")
ax1.set_title("Sisi Lola Voice Fluency Progression",
              fontsize=14, fontweight="bold", pad=12, color="#1A1A1A")
ax1.grid(axis="y", alpha=0.25, zorder=1)
ax1.set_facecolor("#FFFFFC")
for s in ("top","right"): ax1.spines[s].set_visible(False)

# ===== RIGHT: component radar for V4 vs native =====
comps = ["spk_sim", "tone_pres", "diac_acc", "asr_proxy", "natural"]
comp_labels = ["Speaker\nSimilarity", "Tone\nPreservation", "Diacritic\nAccuracy",
               "Intelligibility\n(HNR)", "Naturalness"]
v4_avg = np.array([np.mean([by[f"V4_F5TTS_ep{i}"][c] for i in (1,2,3)]) for c in comps])
nat = np.array([by["W8MQ2_GROUND_TRUTH"][c] for c in comps])

angles = np.linspace(0, 2*np.pi, len(comps), endpoint=False).tolist()
angles += angles[:1]
v4 = v4_avg.tolist() + [v4_avg[0]]
na = nat.tolist() + [nat[0]]

ax2 = fig.add_subplot(1, 2, 2, polar=True)
ax2.set_facecolor("#FFFFFC")
ax2.plot(angles, na, color="#3F8A4D", lw=2.2, label="Native (W8MQ2)")
ax2.fill(angles, na, color="#3F8A4D", alpha=0.18)
ax2.plot(angles, v4, color="#E5894B", lw=2.2, label="V4 F5-TTS (mean)")
ax2.fill(angles, v4, color="#E5894B", alpha=0.25)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(comp_labels, fontsize=9.5)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(["0.25","0.50","0.75","1.0"], fontsize=8, color="#666")
ax2.set_title("V4 vs Native — Component Breakdown",
              fontsize=12, fontweight="bold", pad=18, color="#1A1A1A")
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9.5)
ax2.grid(alpha=0.3)

plt.suptitle("Sisi Lola — Native Yoruba Voice Cloning · Fluency Proof of Concept",
             fontsize=15, fontweight="bold", color="#1A1A1A", y=1.02)
plt.tight_layout()

OUT = Path("/home/user/workspace/sisi_lola_fluency_poc.png")
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#FAFAF7")
print(f"WROTE {OUT}")

# Summary table
print(f"\nNative anchor (W8MQ2): {native:.1f}")
for k, scores in versions.items():
    label = k.replace("\n", " ")
    print(f"{label:45s}  mean={np.mean(scores):5.1f}  min={min(scores):5.1f}  max={max(scores):5.1f}")
