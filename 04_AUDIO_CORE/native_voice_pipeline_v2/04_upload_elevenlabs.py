#!/usr/bin/env python3
"""Upload curated native Yoruba archetype clips to ElevenLabs IVC."""
import os, sys, json, csv
from pathlib import Path
import requests

# The proxy at $HTTPS_PROXY auto-injects the xi-api-key header for api.elevenlabs.io
# So we just need to ensure requests uses the proxy.
PROXIES = {"https": os.environ.get("HTTPS_PROXY"), "http": os.environ.get("HTTPS_PROXY")}
print('Using proxy:', PROXIES['https'][:50] + '...' if PROXIES['https'] else None)
BASE_URL = "https://api.elevenlabs.io/v1"
SRC = Path("/home/user/workspace/voice-clone/ivc_topk")

# Use up to 25 clips, prioritized by similarity (top of file)
files_to_send = []
with open(SRC / "transcripts.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        files_to_send.append(row["filename"])
        if len(files_to_send) >= 25:
            break

print(f"Uploading {len(files_to_send)} clips")
total_dur = 0.0
for fn in files_to_send:
    p = SRC / fn
    print(f"  {fn} ({p.stat().st_size//1024}KB)")

# Build multipart upload
files_payload = []
for fn in files_to_send:
    p = SRC / fn
    files_payload.append(("files", (fn, open(p, "rb"), "audio/wav")))

data = {
    "name": "Sisi Lola Native YO v2",
    "description": "Native Yoruba female voice clone from NaijaVoices archetype cluster — 25 top-similarity clips averaging 0.78+ cosine to Sisi Lola reference. Warm mature big-sis Naija vibe.",
    "labels": json.dumps({
        "accent": "yoruba",
        "language": "yoruba",
        "gender": "female",
        "age": "middle aged",
        "use_case": "narration"
    }),
}

print("\nSending POST to ElevenLabs...")
r = requests.post(
    f"{BASE_URL}/voices/add",
    data=data,
    files=files_payload,
    proxies=PROXIES,
    verify=False,
    timeout=180,
)
print(f"Status: {r.status_code}")
print(r.text[:1000])
if r.status_code == 200:
    voice_id = r.json()["voice_id"]
    print(f"\n=== SUCCESS === voice_id={voice_id}")
    # Save the voice_id
    Path("/home/user/workspace/voice-clone/voice_id.txt").write_text(voice_id)
