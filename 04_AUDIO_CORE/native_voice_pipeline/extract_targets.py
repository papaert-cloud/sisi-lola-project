"""Stream the FLEURS train.tar.gz from HF and extract only target WAVs.
Avoids downloading 1.8GB just to get ~3MB of audio.
"""
import tarfile
import requests
from pathlib import Path

targets_file = Path("/home/user/workspace/voice-clone/targets.txt")
out_dir = Path("/home/user/workspace/voice-clone/references")
out_dir.mkdir(parents=True, exist_ok=True)

# Map: bare filename -> (speaker_id, duration, transcription)
target_map = {}
with targets_file.open() as f:
    for line in f:
        path, spk, dur, *trans = line.strip().split("\t")
        fname = Path(path).name
        target_map[fname] = (spk.strip(), dur, "\t".join(trans))

print(f"Looking for {len(target_map)} target files")
print(f"Targets: {list(target_map.keys())[:3]}...")

url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/yo_ng/audio/train.tar.gz"
print(f"Streaming {url}")

resp = requests.get(url, stream=True, timeout=300)
resp.raise_for_status()

found = 0
total_target = len(target_map)

# Wrap response.raw to be a file-like object for tarfile
with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        bare = Path(member.name).name
        if bare in target_map:
            spk, dur, trans = target_map[bare]
            # Extract this file
            extract_obj = tar.extractfile(member)
            if extract_obj is None:
                continue
            out_name = f"fleurs_yo_spk{spk}_{float(dur):.1f}s.wav"
            out_path = out_dir / out_name
            with out_path.open("wb") as f:
                f.write(extract_obj.read())
            found += 1
            print(f"  [{found}/{total_target}] {out_name}")
            if found >= total_target:
                print("All targets found — stopping early.")
                break

print(f"\nExtracted {found}/{total_target} files to {out_dir}")
