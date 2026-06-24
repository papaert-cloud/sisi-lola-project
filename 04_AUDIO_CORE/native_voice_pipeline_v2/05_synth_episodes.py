#!/usr/bin/env python3
"""Synthesize EP1-3 with the new native Yoruba voice clone (eleven_multilingual_v2)."""
import os, sys
from pathlib import Path
import requests

PROXIES = {"https": os.environ.get("HTTPS_PROXY"), "http": os.environ.get("HTTPS_PROXY")}
BASE_URL = "https://api.elevenlabs.io/v1"
VOICE_ID = "jb6elqjuByGWFmFzxKLY"
SCRIPT_DIR = Path("/home/user/workspace/sisi-lola-project/04_AUDIO_CORE/native_voice_pipeline/scripts")
OUT_DIR = Path("/home/user/workspace/voice-clone/output_v2")
OUT_DIR.mkdir(exist_ok=True)

eps = [
    ("ep1-no-give-up.txt", "ep1-hustle"),
    ("ep2-chop-life.txt", "ep2-food"),
    ("ep3-ede-wa.txt", "ep3-language"),
]

# Best settings for native accent: lower stability = preserves accent quirks
voice_settings = {
    "stability": 0.45,
    "similarity_boost": 0.85,
    "style": 0.35,
    "use_speaker_boost": True,
}

for src, out_name in eps:
    text = (SCRIPT_DIR / src).read_text()
    print(f"\n=== Synthesizing {out_name} ({len(text)} chars) ===")
    
    r = requests.post(
        f"{BASE_URL}/text-to-speech/{VOICE_ID}",
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": voice_settings,
        },
        proxies=PROXIES,
        verify=False,
        timeout=300,
    )
    print(f"Status: {r.status_code}, bytes: {len(r.content)}")
    if r.status_code == 200:
        out_path = OUT_DIR / f"{out_name}.mp3"
        out_path.write_bytes(r.content)
        print(f"  Saved {out_path}")
    else:
        print(f"  ERROR: {r.text[:300]}")

print("\nDone.")
