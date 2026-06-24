"""
Process pending sources:
  1. Download audio with yt-dlp (or requests for podcasts)
  2. VAD-segment into 5-30s clips
  3. Filter: SNR, pitch (female), language conf, similarity to W8MQ2
  4. Accept clips into corpus_v5/clips/

Design: each ingest run processes at most MAX_NEW_PER_RUN sources to stay under
~15 min compute budget on hourly cron. Reference audio kept WAV 24kHz mono for
F5-TTS compatibility.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from manifest import connect, update_source, add_clip, now_iso

ROOT = Path("/home/user/workspace/voice-clone")
RAW_DIR = ROOT / "corpus_v5" / "raw"
CLIPS_DIR = ROOT / "corpus_v5" / "clips"
TRANSCRIPTS_DIR = ROOT / "corpus_v5" / "transcripts"
for d in (RAW_DIR, CLIPS_DIR, TRANSCRIPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

W8MQ2_REF = ROOT / "single_speaker_corpus" / "audio" / "W8MQ2__0_00000__r00368.wav"

MAX_NEW_PER_RUN = 8           # 8h cadence — process more sources per run
TARGET_SR = 24000
MIN_CLIP = 5.0
MAX_CLIP = 30.0
MIN_SNR_DB = 18.0             # was 12 — tighter for refined run
FEMALE_PITCH_MIN = 145        # was 140
FEMALE_PITCH_MAX = 275        # was 280
MIN_SIM_TO_W8MQ2 = 0.40       # was 0.30
MIN_FLUENCY_SCORE = 75.0      # gate clips on calibrated fluency v2 metric


# ---- audio io ----
def load_audio(path, sr=TARGET_SR):
    data, source_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if source_sr != sr:
        # cheap resample with ffmpeg-less linear; acceptable for analysis
        n = int(len(data) * sr / source_sr)
        x_old = np.linspace(0, 1, len(data), endpoint=False)
        x_new = np.linspace(0, 1, n, endpoint=False)
        data = np.interp(x_new, x_old, data).astype("float32")
    return data, sr


def estimate_snr(x):
    """Rough SNR estimate: top 10% energy frames vs bottom 10%."""
    if len(x) == 0:
        return 0.0
    win = 480  # 20ms at 24kHz
    frames = np.array([x[i:i+win] for i in range(0, len(x)-win, win)])
    if len(frames) < 10:
        return 0.0
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    rms = np.sort(rms)
    signal = rms[int(len(rms) * 0.9):].mean()
    noise = rms[:int(len(rms) * 0.1)].mean() + 1e-12
    return float(20 * np.log10(signal / noise))


def estimate_pitch(x, sr=TARGET_SR):
    """Autocorrelation-based F0 mean over voiced frames."""
    win = int(0.04 * sr)  # 40ms
    hop = int(0.02 * sr)
    pitches = []
    for i in range(0, len(x) - win, hop):
        frame = x[i:i+win]
        if (frame ** 2).mean() < 1e-4:
            continue
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[len(frame)-1:]
        # voice range 80-400 Hz -> lag 60..300 samples at 24k
        lag_min, lag_max = sr // 400, sr // 80
        if lag_max >= len(ac):
            continue
        seg = ac[lag_min:lag_max]
        if len(seg) == 0:
            continue
        peak = int(np.argmax(seg)) + lag_min
        if peak > 0:
            pitches.append(sr / peak)
    if not pitches:
        return 0.0
    pitches = np.array(pitches)
    valid = pitches[(pitches > 60) & (pitches < 500)]
    return float(np.median(valid)) if len(valid) else 0.0


def vad_segments(x, sr=TARGET_SR):
    """Energy-based VAD that handles both interview and continuous-speech streams.

    Strategy:
      1. Compute 30ms frame RMS.
      2. Threshold at max(p20 * 1.5, p50 * 0.5) — low enough to keep speech, high
         enough to drop silence.
      3. Merge frames into runs; smooth small gaps (<0.5s).
      4. Split long runs into MAX_CLIP-sized chunks.
      5. If <2 segments found, fall back to a sliding-window chunking of the whole
         audio (useful for continuous radio streams where there's no obvious silence).
    """
    win = int(0.03 * sr)
    hop = win
    if len(x) < win * 10:
        return []
    energy = np.array([(x[i:i+win] ** 2).mean() for i in range(0, len(x) - win, hop)])
    if len(energy) == 0:
        return []
    thr = max(np.percentile(energy, 20) * 1.5, np.percentile(energy, 50) * 0.5, 1e-6)
    speech = energy > thr
    # Smooth: fill gaps < 0.5s
    gap_frames = int(0.5 / 0.03)
    i = 0
    while i < len(speech):
        if speech[i]:
            i += 1; continue
        j = i
        while j < len(speech) and not speech[j]:
            j += 1
        if j - i < gap_frames and i > 0 and j < len(speech):
            speech[i:j] = True
        i = j
    segs = []
    in_seg = False
    start = 0.0
    for i, s in enumerate(speech):
        t = i * 0.03
        if s and not in_seg:
            start = t; in_seg = True
        elif not s and in_seg:
            end = t
            if end - start >= MIN_CLIP:
                cur = start
                while end - cur > MAX_CLIP:
                    segs.append((cur, cur + MAX_CLIP)); cur += MAX_CLIP
                if end - cur >= MIN_CLIP:
                    segs.append((cur, end))
            in_seg = False
    if in_seg:
        end = len(speech) * 0.03
        if end - start >= MIN_CLIP:
            cur = start
            while end - cur > MAX_CLIP:
                segs.append((cur, cur + MAX_CLIP)); cur += MAX_CLIP
            if end - cur >= MIN_CLIP:
                segs.append((cur, end))
    # Fallback: continuous-speech stream — chunk linearly
    if len(segs) < 2:
        dur = len(x) / sr
        chunk = (MIN_CLIP + MAX_CLIP) / 2  # ~17s
        segs = []
        t = 0.0
        while t + MIN_CLIP <= dur:
            segs.append((t, min(t + chunk, dur)))
            t += chunk
    return segs


# ---- similarity (cheap MFCC + cosine; Resemblyzer would need more disk) ----
def mfcc_lite(x, sr=TARGET_SR, n_mfcc=20):
    """13-D mean MFCC via scipy if available, else log-mel."""
    try:
        from scipy.fftpack import dct
        n_fft = 512
        hop = 160
        mel_bins = 40
        # log-mel
        frames = np.array([x[i:i+n_fft] for i in range(0, len(x)-n_fft, hop)])
        if len(frames) == 0:
            return np.zeros(n_mfcc)
        spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
        # triangular mel filterbank
        mel_min = 2595 * np.log10(1 + 0 / 700)
        mel_max = 2595 * np.log10(1 + (sr/2) / 700)
        m_pts = np.linspace(mel_min, mel_max, mel_bins + 2)
        f_pts = 700 * (10 ** (m_pts / 2595) - 1)
        bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
        fb = np.zeros((mel_bins, n_fft // 2 + 1))
        for m in range(1, mel_bins + 1):
            l, c, r = bins[m-1], bins[m], bins[m+1]
            if c > l:
                fb[m-1, l:c] = (np.arange(l, c) - l) / (c - l)
            if r > c:
                fb[m-1, c:r] = (r - np.arange(c, r)) / (r - c)
        mel = np.log(fb @ spec.T + 1e-8)
        cep = dct(mel, axis=0, norm="ortho")[:n_mfcc]
        return cep.mean(axis=1)
    except Exception as e:
        print(f"[mfcc] fallback (no scipy): {e}", file=sys.stderr)
        return np.array([x.std(), np.mean(np.abs(x)), x.min(), x.max(),
                         np.median(x), np.percentile(x, 25), np.percentile(x, 75),
                         len(x)], dtype="float32")


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Load W8MQ2 archetype vector once
def _w8mq2_vec_cache():
    cache = ROOT / "corpus_v5" / "manifests" / "w8mq2_vec.npy"
    if cache.exists():
        return np.load(cache)
    x, _ = load_audio(W8MQ2_REF)
    v = mfcc_lite(x)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, v)
    return v


W8MQ2_VEC = None


# ---- download ----
def download_youtube(url, out_dir):
    """Use yt-dlp to grab audio as wav. Returns path or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")
    try:
        r = subprocess.run(
            ["yt-dlp", "-q", "-f", "bestaudio", "-x", "--audio-format", "wav",
             "--audio-quality", "0", "--no-playlist",
             "--max-filesize", "120M", "--max-downloads", "1",
             "-o", out_tmpl, url],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            return None, r.stderr[-500:]
        # find produced wav
        for p in out_dir.glob("*.wav"):
            return p, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    return None, "no_output"


def download_podcast(url, dest):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        return dest, None
    except Exception as e:
        return None, str(e)


def capture_radio(url, dest_wav, seconds=90):
    """Capture N seconds of a live stream as 24kHz mono wav."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-user_agent", "Mozilla/5.0",
             "-i", url, "-t", str(int(seconds)),
             "-ac", "1", "-ar", str(TARGET_SR),
             "-c:a", "pcm_s16le", str(dest_wav)],
            capture_output=True, text=True, timeout=seconds + 30,
        )
        if r.returncode != 0 or not dest_wav.exists():
            return None, (r.stderr or "ffmpeg_failed")[-300:]
        return dest_wav, None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)


# ---- main per-source processing ----
def process_source(conn, src):
    src_id = src["source_id"]
    print(f"[process] {src_id} ({src['source_type']}) {src['title']!r}", flush=True)
    raw_dest_dir = RAW_DIR / src_id.replace(":", "_").replace("/", "_")
    if src["source_type"] == "youtube":
        path, err = download_youtube(src["url"], raw_dest_dir)
    elif src["source_type"] == "radio":
        raw_dest_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dest_dir / "capture.wav"
        secs = src.get("duration_sec") or 90
        path, err = capture_radio(src["url"], path, seconds=int(secs))
    else:
        raw_dest_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dest_dir / "audio.mp3"
        path, err = download_podcast(src["url"], path)
    if path is None:
        update_source(conn, src_id, status="rejected", reject_reason=f"download_failed:{err[:80]}")
        return 0, 0
    update_source(conn, src_id, status="downloaded", raw_path=str(path))

    x, sr = load_audio(path)
    segs = vad_segments(x, sr)
    if not segs:
        update_source(conn, src_id, status="rejected", reject_reason="no_speech")
        try: path.unlink(); shutil.rmtree(raw_dest_dir, ignore_errors=True)
        except Exception: pass
        return 0, 0

    global W8MQ2_VEC
    if W8MQ2_VEC is None:
        W8MQ2_VEC = _w8mq2_vec_cache()

    accepted = 0
    new_clips = 0
    for (s, e) in segs[:80]:  # cap clips per source
        new_clips += 1
        seg_x = x[int(s*sr):int(e*sr)]
        if len(seg_x) < MIN_CLIP * sr:
            continue
        snr = estimate_snr(seg_x)
        pitch = estimate_pitch(seg_x, sr)
        vec = mfcc_lite(seg_x, sr)
        sim = cosine(vec, W8MQ2_VEC)
        is_female = int(FEMALE_PITCH_MIN <= pitch <= FEMALE_PITCH_MAX)

        reject = None
        if snr < MIN_SNR_DB:
            reject = f"low_snr_{snr:.1f}"
        elif not is_female:
            reject = f"pitch_{pitch:.0f}_not_female"
        elif sim < MIN_SIM_TO_W8MQ2:
            reject = f"low_sim_{sim:.2f}"

        clip_id = f"{src_id.replace(':','_')}__{int(s*1000):07d}_{int(e*1000):07d}"
        clip_path = None
        fluency = None
        if reject is None:
            clip_path = CLIPS_DIR / f"{clip_id}.wav"
            sf.write(clip_path, seg_x, sr)
            # Fluency scoring gate (calibrated v2)
            try:
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).parent))
                from fluency_metric_v2 import score_file
                fr = score_file(clip_path)
                fluency = fr.score
                if fluency < MIN_FLUENCY_SCORE:
                    reject = f"low_fluency_{fluency:.1f}"
                    clip_path.unlink(missing_ok=True)
                    clip_path = None
            except Exception as _fe:
                print(f"[fluency] scoring failed for {clip_id}: {_fe}", file=sys.stderr)
            if reject is None:
                accepted += 1

        add_clip(
            conn,
            clip_id=clip_id,
            source_id=src_id,
            start_sec=float(s),
            end_sec=float(e),
            duration_sec=float(e - s),
            speaker_label=None,
            is_female=is_female,
            pitch_mean_hz=float(pitch),
            snr_db=float(snr),
            lang="yo",  # placeholder; whisper step would refine
            lang_conf=0.0,
            text=None,
            sim_to_w8mq2=float(sim),
            clip_path=str(clip_path) if clip_path else None,
            accepted=1 if reject is None else 0,
            reject_reason=reject,
        )

    status = "accepted" if accepted > 0 else "rejected"
    update_source(
        conn, src_id, status=status,
        reject_reason=None if accepted else "no_clips_passed_gates",
    )
    # cleanup raw if nothing accepted
    if accepted == 0:
        try:
            path.unlink(); shutil.rmtree(raw_dest_dir, ignore_errors=True)
        except Exception:
            pass
    print(f"[process] {src_id} segments={len(segs)} new_clips={new_clips} accepted={accepted}",
          flush=True)
    return new_clips, accepted


def run_processing(max_sources=MAX_NEW_PER_RUN):
    conn = connect()
    # Always process radio captures (they're the most reliable source);
    # plus top tier-A non-radio sources up to the budget.
    radio_pending = [dict(r) for r in conn.execute(
        "SELECT * FROM sources WHERE status='pending' AND source_type='radio'"
    ).fetchall()]
    other_pending = [dict(r) for r in conn.execute(
        "SELECT * FROM sources WHERE status='pending' AND source_type != 'radio' "
        "ORDER BY CASE tier WHEN 'A' THEN 0 WHEN 'B' THEN 1 ELSE 2 END, ingested_at LIMIT ?",
        (max_sources,),
    ).fetchall()]
    pending = radio_pending + other_pending
    total_new = 0
    total_acc = 0
    for src in pending:
        try:
            n, a = process_source(conn, src)
            total_new += n
            total_acc += a
        except Exception as e:
            print(f"[process] FAILED {src['source_id']}: {e}", file=sys.stderr)
            update_source(conn, src["source_id"], status="rejected",
                          reject_reason=f"exception:{str(e)[:80]}")
    print(f"[process] sources_processed={len(pending)} new_clips={total_new} "
          f"accepted={total_acc}")
    return len(pending), total_new, total_acc


if __name__ == "__main__":
    run_processing()
