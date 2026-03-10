"""
test_audio.py
=============
Offline test harness for the audio proctoring pipeline.

Usage:
    python test_audio.py                          # processes inputStage1/ folder
    python test_audio.py inputStage1/             # specific folder
    python test_audio.py session_recording.wav            # single file
    python test_audio.py inputStage1/ --enroll enrollment.wav
    python test_audio.py onlyme.wav --verbose

lip_activity=True is passed on every chunk so GHOST_VOICE is suppressed and
focus is on EXTRA_SPEAKER / VOICE_MISMATCH. No audio clips are written.

--verbose  Print per-embedding sim scores as they happen.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from config import AUDIO_SR, AUDIO_CHUNK
from core.audio_proctoring import ProctorSession, CheatType

# ── Defaults ──────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).parent
_DEFAULT_INPUT = _ROOT / "inputStage1"
_DEFAULT_ENROLL = _ROOT / "enrollment.wav"

_LABELS = {
    CheatType.IMPERSONATION:  "IMPERSONATION  — voice ≠ enrolled, lips still",
    CheatType.GHOST_VOICE:    "GHOST_VOICE    — enrolled voice, lips not moving",
    CheatType.EXTRA_SPEAKER:  "EXTRA_SPEAKER  — 2+ distinct speaker clusters",
    CheatType.VOICE_MISMATCH: "VOICE_MISMATCH — voice does not match enrolled",
}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _load_wav(path: str | Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return _resample(audio, sr, target_sr)


def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=sr,
                           stationary=True, prop_decrease=0.85).astype(np.float32)


# ── Verbose patch ─────────────────────────────────────────────────────────────

def _patch_verbose(session: ProctorSession) -> list[dict]:
    """Intercept embed() after enrollment to log per-embedding diagnostics."""
    log: list[dict] = []
    original_embed = session._embedder.embed

    def _patched(audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
        emb = original_embed(audio, sr)
        if emb is not None and session._profile is not None:
            enroll = session._profile.centroid
            sim = float(np.dot(
                emb / (np.linalg.norm(emb) + 1e-8),
                enroll / (np.linalg.norm(enroll) + 1e-8),
            ))
            t = session._chunk_count * AUDIO_CHUNK / AUDIO_SR
            m, s = divmod(int(t), 60)
            print(f"  [embed {m:02d}:{s:02d}]  sim={sim:.4f}  "
                  f"n_speakers={session._tracker.n_speakers}")
            log.append({"sim": sim, "t": t})
        return emb

    session._embedder.embed = _patched
    return log


# ── Single-file test ──────────────────────────────────────────────────────────

def _run_file(
    wav_path: Path,
    session: ProctorSession,
    verbose: bool,
    embed_log: list[dict],
) -> tuple[list, float]:
    """
    Run one WAV file through the session (which is already enrolled).
    Resets session state between files but keeps enrollment.
    Returns (events, duration_s).
    """
    session.reset()

    try:
        audio = _denoise(_load_wav(wav_path, AUDIO_SR), AUDIO_SR)
    except Exception as e:
        print(f"  [Load ERROR] {e}")
        return [], 0.0

    duration_s   = len(audio) / AUDIO_SR
    total_chunks = len(audio) // AUDIO_CHUNK

    print(f"\n  File: {wav_path.name}  ({duration_s:.1f}s, {total_chunks} chunks)")
    print(f"  {'─'*52}")

    if verbose:
        print(f"  [Verbose] per-embedding log:")

    all_events: list = []

    for i in range(total_chunks):
        chunk = audio[i * AUDIO_CHUNK : (i + 1) * AUDIO_CHUNK].copy()
        for ev in session.push(chunk, lip_activity=True):
            all_events.append(ev)
            label = _LABELS.get(ev.event_type, ev.event_type.name)
            m, s = divmod(int(ev.timestamp_s), 60)
            print(f"  ⚠ [{m:02d}:{s:02d}]  {label}")
            print(f"         conf={ev.confidence:.0%}  "
                  f"score={ev.verify_score:.3f}  "
                  f"speakers={ev.n_speakers}")

    summary = session.get_summary()
    print(f"\n  {'─'*52}")
    print(f"  Speech: {summary['total_speech_s']:.1f}s  |  "
          f"Speakers: {summary['n_speakers_detected']}  |  "
          f"Score mean: {summary['verify_score_mean']:.4f}  |  "
          f"Events: {len(all_events)}")

    if all_events:
        print(f"  Timestamps:")
        for ev in all_events:
            m, s = divmod(int(ev.timestamp_s), 60)
            print(f"    [{m:02d}:{s:02d}]  {ev.event_type.name:<20}  "
                  f"conf={ev.confidence:.0%}  score={ev.verify_score:.3f}")

    if verbose and embed_log:
        sims = [e["sim"] for e in embed_log]
        print(f"  Embed stats: mean={np.mean(sims):.4f}  "
              f"min={np.min(sims):.4f}  max={np.max(sims):.4f}  "
              f"std={np.std(sims):.4f}")
        embed_log.clear()

    return all_events, duration_s


# ── Main entry ────────────────────────────────────────────────────────────────

def run(input_path: Path, enroll_wav: Path, verbose: bool) -> None:
    # Resolve list of files to process
    if input_path.is_dir():
        wav_files = sorted(input_path.glob("*.wav"))
        if not wav_files:
            print(f"No .wav files found in {input_path}")
            sys.exit(1)
        mode = "dir"
    elif input_path.is_file() and input_path.suffix.lower() == ".wav":
        wav_files = [input_path]
        mode = "file"
    else:
        print(f"ERROR: '{input_path}' is not a .wav file or a directory.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Audio Proctoring — Offline Test")
    print(f"  Mode       : {'directory' if mode == 'dir' else 'single file'}")
    print(f"  Input      : {input_path}")
    print(f"  Enrollment : {enroll_wav}")
    print(f"  Files      : {len(wav_files)}")
    print(f"{'='*60}")

    # Enroll once
    session = ProctorSession(sample_rate=AUDIO_SR)
    try:
        enroll_audio = _denoise(_load_wav(enroll_wav, AUDIO_SR), AUDIO_SR)
        session.enroll(enroll_audio, sr=AUDIO_SR)
        print(f"\n[Enroll] OK — {len(enroll_audio)/AUDIO_SR:.2f}s @ {AUDIO_SR}Hz")
    except FileNotFoundError:
        print(f"[Enroll] ERROR: '{enroll_wav}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"[Enroll] ERROR: {e}")
        sys.exit(1)

    embed_log = _patch_verbose(session) if verbose else []

    # Process files
    t0 = time.monotonic()
    overall: list[tuple[str, list, float]] = []   # (name, events, duration)

    for wav_path in wav_files:
        events, duration = _run_file(wav_path, session, verbose, embed_log)
        overall.append((wav_path.name, events, duration))

    elapsed = time.monotonic() - t0

    # Overall summary (only shown in dir mode with multiple files)
    if mode == "dir" and len(wav_files) > 1:
        print(f"\n{'='*60}")
        print(f"  OVERALL — {len(wav_files)} files  ({elapsed:.1f}s processing)")
        print(f"{'='*60}")
        print(f"  {'File':<35} {'Speech':>7}  {'Events':>6}")
        print(f"  {'─'*52}")
        total_events = 0
        for name, events, dur in overall:
            flag = "⚠" if events else "✓"
            print(f"  {flag} {name:<33} {dur:>6.1f}s  {len(events):>6}")
            total_events += len(events)
        print(f"  {'─'*52}")
        print(f"  Total cheat events: {total_events}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline audio proctoring test")
    parser.add_argument(
        "input", nargs="?", default=str(_DEFAULT_INPUT),
        help=f"WAV file or folder of WAVs (default: {_DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--enroll", default=str(_DEFAULT_ENROLL),
        help=f"Enrollment WAV (default: {_DEFAULT_ENROLL})"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-embedding sim scores"
    )
    args = parser.parse_args()

    run(Path(args.input), Path(args.enroll), verbose=args.verbose)
