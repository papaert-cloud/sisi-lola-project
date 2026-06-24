# Sisi Lola Voice — Roll-Plan & Model Versioning Strategy

**Owner:** BAMG Studio · **Updated:** 2026-06-24
**Goal:** Reach native-tier Yoruba fluency (composite ≥ 97 on `fluency_metric_v2`) for Sisi Lola while preserving every intermediate generation as a reproducible, comparable artifact.

---

## 1. Version Registry

Every generation is a tagged, immutable artifact with a manifest, an audio bundle, and a fluency score.

| Tag | Engine | Reference Strategy | Mean Fluency | Status | Notes |
|---|---|---|---|---|---|
| `v1` | ElevenLabs (English VO mix) | — | n/a | archived | Pre-Yoruba baseline |
| `v2` | ElevenLabs (Yorunglish) | single voice | 92.4 | retired | NATIVE_INDISTINGUISHABLE but not pure Yoruba |
| `v3` | ElevenLabs (pure Yoruba forced) | single voice | 89.3 | retired | Engine had no native Yoruba support |
| `v4` | F5-TTS naijaml/f5-tts-yoruba | 100% W8MQ2 single ref | 91.0 (EP2 = 95.3) | shipped (preview) | Best episode = EP2 Chop Life |
| **`v5`** | **F5-TTS naijaml/f5-tts-yoruba** | **50% W8MQ2 + 25% HZM4L + 25% V4-EP2 anchor** | **target ≥ 94** | **active** | Female-blend with native HZM4L; EP2 anchor scaled back |
| `v6` (planned) | F5-TTS fine-tuned on combined W8MQ2+HZM4L corpus | LoRA / checkpoint merge | target ≥ 97 | planned | Triggered after corpus reaches 30 min |
| `v7` (planned) | F5-TTS fine-tune routed to Replicate/Modal | full retrain | target ≥ 99 | planned | Once we hit corpus ≥ 60 min, native-only data |

**Tagging convention:** every artifact is named `sisi-lola/<tag>/<episode>__<engine>__<ts>.{wav,mp3}` and pinned to a git tag `voice/<tag>` plus a Dropbox snapshot folder `SISI_LOLA_PROTOTYPE/<tag>/`.

---

## 2. Three-Track Experiment Strategy

We run three concurrent tracks per major version. Each track produces its own manifest so we can A/B without losing the lineage.

### Track A — **Combine** (multi-reference blend)
- Multiple speakers blended by weighted reference sampling at synthesis time.
- Currently active for v5: W8MQ2 + HZM4L + V4-EP2 anchor.
- Cheap to iterate (no training), works on existing checkpoints.
- Trade-off: voice identity can drift if weights are off.

### Track B — **Isolate** (per-episode / per-component A/B)
- Hold reference constant per episode, vary one parameter at a time:
  - reference clip duration (5s vs 8s vs 12s)
  - pitch normalization on/off
  - F5-TTS `nfe_step` sweeps (16, 32, 64)
- Used to find the sweet spot on a single dimension before propagating to Combine.
- Episodes EP1 / EP2 / EP3 each act as a fixed test bed.

### Track C — **Finetune** (full or LoRA training on the corpus)
- Use the curated single-speaker corpus once it crosses 30 min.
- Two finetune routes (decided per run cost & latency):
  1. **Replicate** — `replicate.com/naijaml/f5-tts-yoruba` train endpoint when available; fast iteration, pay-per-run.
  2. **Modal** — `modal.com` GPU job for full retrain when we want full control of training script + checkpoints.
  3. **Hugging Face Spaces / AutoTrain** — fallback for free-tier sanity checks.
- Each finetune produces a `.pt` checkpoint pinned to git LFS + Dropbox archive.

---

## 3. Hyper-parameter & Reference-Weight Policy

The active policy is defined in [`V5_REFERENCE_POLICY.yaml`](./V5_REFERENCE_POLICY.yaml). Summary:

| Reference | Weight | Role | Source clip |
|---|---|---|---|
| W8MQ2 | 50% | primary native female | `W8MQ2__0_00000__r00368.wav` (8.76s, score 100.0) |
| HZM4L | 25% | secondary native female blend | `HZM4L__0_00003__r01195.wav` (3.1s, score 99.9) |
| V4 EP2 Chop Life | 25% | episodic style anchor (scaled back per user) | `output_v4/ep2_chop_life.wav` |

**Synthesis-time gates** (applied by `scripts/11_f5tts_yoruba_synth.py`):
- composite ≥ 75 → accept; else regenerate with alternate reference
- speaker similarity to W8MQ2 ≥ 0.40
- pitch drift from reference ≤ 30 Hz

---

## 4. Corpus Pipeline (already shipped)

- **Cron**: `4d3447fe` runs every 8h (UTC 04:57 / 12:57 / 20:57).
- **Sources** (refined v2): BBC Yorùbá, TVC News, Afrikana Radio, Bond FM Lagos, Splash FM Ibadan.
- **Quality gates** (`process.py`): SNR ≥ 18 dB · female pitch 145–275 Hz · MFCC sim ≥ 0.40 to W8MQ2 · Yoruba lang conf ≥ 0.80 · fluency ≥ 75.
- **Output**: `corpus_v5/clips/` + manifest `corpus_v5/manifests/fluency_scores_v2.jsonl`.
- **Slack reporting**: every run posts a summary DM with new minutes added, accepted/rejected counts, and mean fluency.

Corpus growth thresholds trigger version bumps:
- **≥ 30 min total** → eligible for v6 LoRA finetune
- **≥ 60 min native-only** → eligible for v7 full retrain

---

## 5. Storage & Source-of-Truth

| Artifact | Local | Remote |
|---|---|---|
| Code & manifests | `/home/user/workspace/voice-clone/` + `sisi-lola-project/` git repo | GitHub `papaert-cloud/sisi-lola-project` (PR #3, branch `feat/native-voice-clone-2026-06`) |
| Audio outputs | `voice-clone/output_v4/`, `voice-clone/output_v5/` | Dropbox `Sisi_Lola/SISI_LOLA_PROTOTYPE/<tag>/` |
| Reference clips | `voice-clone/single_speaker_corpus/audio/` | Dropbox `SISI_LOLA_PROTOTYPE/references/` |
| Checkpoints (.pt) | `voice-clone/f5tts_yoruba/` | git LFS + Dropbox `SISI_LOLA_PROTOTYPE/checkpoints/` |
| Fluency reports | `voice-clone/corpus_v5/manifests/` | Dropbox `SISI_LOLA_PROTOTYPE/fluency/` |

Versioning rules:
- **No artifact is ever overwritten.** New version → new folder.
- **Every release tag** has a `manifest.json` listing the references used, the seed, and the composite scores for each episode.
- **Git LFS** tracks `*.pt`, `*.wav`, `*.mp3` over 50 MB; small previews stay in the regular tree.

---

## 6. Decision Cadence

| Cadence | Action |
|---|---|
| Every 8h | Corpus ingest + Slack summary (automatic) |
| Per generation | Score with `fluency_metric_v2`, log to manifest |
| When mean fluency ≥ 94 | Promote candidate to a shipped tag, snapshot to Dropbox |
| When corpus ≥ 30 min | Trigger v6 LoRA finetune on Replicate/Modal |
| When v6 ≥ 97 mean | Promote to "native-tier shipped", begin v7 planning |

---

## 7. Open Items (this segment)

- [ ] Wire reference-weight sampler into `scripts/11_f5tts_yoruba_synth.py`
- [ ] Generate v5 EP1/EP2/EP3 with the new policy, score, and publish
- [ ] Connect Replicate + Hugging Face (auth pending)
- [ ] Push branch `feat/native-voice-clone-2026-06` and update PR #3
- [ ] Snapshot v4 outputs + W8MQ2 reference + fluency PoC to Dropbox `SISI_LOLA_PROTOTYPE/`
