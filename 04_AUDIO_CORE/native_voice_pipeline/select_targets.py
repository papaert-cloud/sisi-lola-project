"""Find best FLEURS Yoruba female samples and write a target list."""
import csv
from pathlib import Path

tsv_path = Path("/home/user/workspace/voice-clone/yo_ng_train.tsv")
SAMPLE_RATE = 16000  # FLEURS native sample rate

female_targets = []
with tsv_path.open(encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) < 6:
            continue
        speaker_id, fname, transcription, _norm, _phon, num_samples_or_dur = row[0], row[1], row[2], row[3], row[4], row[5]
        gender = row[6] if len(row) > 6 else ""
        if gender.strip().upper() != "FEMALE":
            continue
        try:
            num_samples = int(num_samples_or_dur)
        except ValueError:
            continue
        duration = num_samples / SAMPLE_RATE
        if 10 <= duration <= 25:
            female_targets.append({
                "speaker_id": speaker_id,
                "filename": fname,
                "transcription": transcription,
                "duration": duration,
            })

# Sort by duration (longest = most clone material), then deduplicate by speaker
female_targets.sort(key=lambda x: -x["duration"])

# Pick diverse speakers: 1 sample per speaker, up to 8
seen_speakers = set()
top = []
for t in female_targets:
    if t["speaker_id"] in seen_speakers:
        continue
    seen_speakers.add(t["speaker_id"])
    top.append(t)
    if len(top) >= 8:
        break

print(f"Total female samples (10-25s): {len(female_targets)}")
print(f"\nTop 8 diverse-speaker female candidates:")
for i, t in enumerate(top):
    print(f"  {i+1}. spk{t['speaker_id']:>3} | {t['filename']} | {t['duration']:.1f}s | '{t['transcription'][:55]}...'")

# Save target filenames for tar extraction
out = Path("/home/user/workspace/voice-clone/targets.txt")
with out.open("w") as f:
    for t in top:
        f.write(f"yo_ng/audio/train/{t['filename']}\t{t['speaker_id']}\t{t['duration']:.1f}\t{t['transcription']}\n")
print(f"\nSaved targets to {out}")
