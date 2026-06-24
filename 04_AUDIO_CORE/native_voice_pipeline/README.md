# Sisi Lola — Native Voice Cloning Pipeline

> **Goal**: produce Sisi Lola episodes in **authentic native Yoruba/Nigerian Pidgin English** ("Yorunglish"), without relying on third-party paid voice services.

This pipeline is **enterprise-owned** — every model and asset lives inside this repository. No external API quotas. No per-character billing. The same script will reproduce the voice exactly.

## Architecture

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ FLEURS Yoruba (HF)   │    │ Sisi Lola Native     │    │ XTTS-v2 (Coqui)      │
│ female speakers      │ -> │ Reference Clip       │ -> │ zero-shot cloner     │
│ (8 speakers, ~24s)   │    │ (75s, 22 kHz mono)   │    │ multilingual + Naija │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
                                                                  │
                                ┌─────────────────────────────────┘
                                ▼
                ┌──────────────────────────┐    ┌──────────────────────┐
                │ Episode Scripts          │    │ Episode WAVs / MP3s  │
                │ (Pidgin + Yoruba text)   │ -> │ Native-voice audio   │
                └──────────────────────────┘    └──────────────────────┘
                                                          │
                                                          ▼
                                                ┌──────────────────────┐
                                                │ Ken-Burns Video      │
                                                │ (keyframes + audio)  │
                                                └──────────────────────┘
```

## Files

| Path | Purpose |
| --- | --- |
| `select_targets.py` | Scans the FLEURS Yoruba `train.tsv` for native-female samples 10–25 s long; deduplicates by speaker and picks the top 8 by duration. |
| `extract_targets.py` | Streams `train.tar.gz` from HuggingFace and extracts **only** the 8 target WAVs (≈ 12 MB) without downloading the full 1.8 GB. |
| `references/` | The 8 native-speaker clips plus the combined `sisi_lola_native_yoruba_reference.wav` (75 s, three speakers concatenated with 0.3 s gaps, loudness-normalised to −18 LUFS). |
| `synth_chunked.py` | Runs XTTS-v2 in CPU mode, splits scripts into ~140-char chunks, zero-shot-clones the native reference voice per chunk, and ffmpeg-concatenates the results into one WAV + MP3 per episode. |
| `scripts/` | Episode scripts in Yorunglish with full Yoruba diacritics. |
| `output/` | Final native-voice audio (`ep1-hustle.mp3`, `ep2-food.mp3`, `ep3-language.mp3`). |
| `renderer/render_videos.py` | Re-renders each episode video with a Ken-Burns crossfade across the two existing keyframes synchronized to the new audio length. |

## Reproducing

```bash
# 1. Install dependencies (uses the system Python 3.11+)
pip install coqui-tts torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu
pip install soundfile librosa requests

# 2. Select native female targets from FLEURS metadata
python 04_AUDIO_CORE/native_voice_pipeline/select_targets.py

# 3. Download the actual WAVs from HuggingFace (streamed, ~12 MB instead of 1.8 GB)
python 04_AUDIO_CORE/native_voice_pipeline/extract_targets.py

# 4. Build the 75-second combined reference clip
ffmpeg -y \
  -i references/fleurs_yo_spk394_25.0s.wav \
  -i references/fleurs_yo_spk1422_24.8s.wav \
  -i references/fleurs_yo_spk46_24.7s.wav \
  -filter_complex "[0:a]aresample=22050,aformat=channel_layouts=mono,loudnorm=I=-18:TP=-2[a0]; \
                   [1:a]aresample=22050,aformat=channel_layouts=mono,loudnorm=I=-18:TP=-2[a1]; \
                   [2:a]aresample=22050,aformat=channel_layouts=mono,loudnorm=I=-18:TP=-2[a2]; \
                   aevalsrc=0:d=0.3:s=22050[gap1]; \
                   aevalsrc=0:d=0.3:s=22050[gap2]; \
                   [a0][gap1][a1][gap2][a2]concat=n=5:v=0:a=1[out]" \
  -map "[out]" -ar 22050 -ac 1 -c:a pcm_s16le \
  references/sisi_lola_native_yoruba_reference.wav

# 5. Synthesize all three episodes (~25 minutes on 8-core CPU)
python 04_AUDIO_CORE/native_voice_pipeline/synth_chunked.py

# 6. Render videos
python 04_AUDIO_CORE/native_voice_pipeline/renderer/render_videos.py
```

## Quality Tuning

- **Reference length matters.** Coqui XTTS-v2 trims to the first 30 seconds — keep the most expressive native segments first.
- **Loudness normalisation** (`loudnorm=I=-18:TP=-2`) keeps multi-speaker references coherent.
- **Chunk size of ~140 chars** is the sweet spot for CPU inference: smaller chunks lose prosody, larger chunks slow down quadratically because of GPT-2 attention.
- **`split_sentences=False`** inside `tts_to_file` is intentional — we are pre-chunked.

## Upgrade Path

1. **GPU inference via Modal/Replicate** — the `08_MLOPS_PIPELINE` and `12_MODEL_SYNC` folders already wrap Modal/Replicate clients. Drop XTTS into the existing Modal app to take inference from ~3× realtime (CPU) down to ~0.05× (T4 GPU).
2. **Fine-tune XTTS on a larger Naija dataset** — add Naijavoices, AfriSpeech-200, or Common Voice Yoruba to `ml_training/data/voice_samples/` and use `08_MLOPS_PIPELINE/training/train_xtts_tts.py`.
3. **Premium fallback** — ElevenLabs Multilingual v2 with Instant Voice Clone using the same `sisi_lola_native_yoruba_reference.wav`. Requires Starter plan or higher.

## License

The FLEURS dataset is released under CC-BY-4.0 by Google.
XTTS-v2 weights are released under the Coqui Public Model License (CPML) — non-commercial.
For commercial deployment, fine-tune XTTS yourself on permissively-licensed data (Naijavoices, Common Voice) or licence the Coqui XTTS commercial weights.
