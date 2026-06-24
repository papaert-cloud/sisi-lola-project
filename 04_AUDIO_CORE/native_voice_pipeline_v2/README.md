# Native Voice Pipeline v2 — Real Data + ElevenLabs PVC

**Iteration goal**: Move from XTTS-v2 zero-shot (v1) to a true native-fluency clone using real native speaker data and ElevenLabs voice cloning.

## What changed from v1

| Aspect | v1 (XTTS-v2) | v2 (ElevenLabs) |
|---|---|---|
| Engine | Self-hosted Coqui XTTS-v2 (CPU) | ElevenLabs Multilingual v2 |
| Reference data | 75s, 3 FLEURS Yoruba speakers concat | 25 clips, 8 min, archetype-clustered NaijaVoices speakers |
| Source | FLEURS (clean read speech) | NaijaVoices (85h TTS-quality female native speech) |
| Selection | Manual ID picks | Speaker-embedding cosine similarity to Sisi Lola reference |
| Speed | ~30-105s per 140-char chunk on 8 vCPU | Single ~75-80s episode in <30s API call |
| Yoruba diacritics | Reference-driven, sometimes warped | Multilingual v2 native handling |
| Cost | Free (self-hosted) | $22/mo Creator tier, 131K chars/mo |

## Pipeline overview

```
NaijaVoices HF dataset (parquet shards)
        │
        ▼  01_extract_corpus.py
   33.9 min of female 30-over Yoruba speech
   (115 unique speakers, in final_corpus/)
        │
        ▼  03_pick_sisi_archetype.py
   8.1 min, 25 clips closest to Sisi Lola
   archetype reference (cosine sim 0.78-0.84)
        │
        ▼  04_upload_elevenlabs.py
   ElevenLabs IVC voice clone created
   voice_id: jb6elqjuByGWFmFzxKLY
        │
        ▼  05_synth_episodes.py
   EP1-3 native audio
   (eleven_multilingual_v2, settings:
    stability=0.45, similarity=0.85, style=0.35)
        │
        ▼  06_render_videos.py
   1080p Ken-Burns videos with native audio
```

## Files

- **`01_extract_corpus.py`** — Streams NaijaVoices parquet shards from HuggingFace, filters `gender=female` + `age_range=30-over`, applies clip-level QC (RMS, peak, duration), writes WAV + TSV
- **`02_cluster_and_select.py`** — Resemblyzer speaker embeddings + Agglomerative cosine clustering to find acoustically coherent group
- **`03_pick_sisi_archetype.py`** — Ranks all corpus clips by cosine similarity to existing Sisi Lola reference, takes top-K until 8 min budget hit, builds concatenated reference
- **`04_upload_elevenlabs.py`** — Multipart upload of 25 top-similarity WAVs to ElevenLabs `/v1/voices/add`
- **`05_synth_episodes.py`** — TTS via `eleven_multilingual_v2` with accent-preserving voice settings
- **`06_render_videos.py`** — ffmpeg Ken-Burns zoompan from avatar PNG synced to MP3 duration

## Voice settings rationale

| Setting | Value | Why |
|---|---|---|
| stability | 0.45 | Low-to-mid = preserves native accent quirks rather than smoothing them away |
| similarity_boost | 0.85 | High = stay close to reference speakers' timbre |
| style | 0.35 | Moderate = some expressive variation without losing accent |
| use_speaker_boost | true | Reinforces speaker identity in long passages |

## Data source: NaijaVoices

- HuggingFace: `EYEDOL/naija-voices-yoruba-split_0-X` (X=0..5)
- 85h TTS-quality female native Yoruba (originally `David-A-Amoo/naijavoices_dataset_85_hours_tts_best`)
- 48kHz, transcribed, gender + age_range labels
- Used because podcast scraping (yt-dlp) is blocked from server IPs and license-clean

## Episode durations (v2 vs v1)

| Episode | v1 (XTTS) | v2 (ElevenLabs) |
|---|---|---|
| EP1 No Give Up | 123.4s | 74.5s |
| EP2 Chop Life | 128.2s | 80.9s |
| EP3 Èdè Wa | 131.8s | 79.0s |

v2 is ~40% shorter because ElevenLabs paces more naturally — no over-elongation. Same scripts.

## Reproducing

```bash
# 1. Extract corpus (~30 min runtime, ~5GB HF cache, ~200MB output)
python3 01_extract_corpus.py

# 2. Cluster + pick archetype (~5 min CPU)
python3 03_pick_sisi_archetype.py

# 3. Upload to ElevenLabs (requires ELEVENLABS_API_KEY)
python3 04_upload_elevenlabs.py

# 4. Synthesize episodes
python3 05_synth_episodes.py

# 5. Render videos
python3 06_render_videos.py
```

## Future iterations

- **v3 candidate**: Submit the larger 30-min cluster as PVC (Professional Voice Clone) instead of IVC, accepting that it'll be a blended archetype voice
- **v4 candidate**: Fine-tune XTTS-v2 on the NaijaVoices corpus on rented GPU (Modal/Replicate) — enterprise-owned weights
- **v5 candidate**: Add Pidgin code-switching by including Common Voice Nigerian Pidgin corpus in the cluster
