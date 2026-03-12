from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ProofWriter:
    """
    Captures proof evidence around proctoring alert events.

    Three proof types:
      image  — single JPEG frame (objects: phone, book, headphone, earbud)
      video  — 5s .mp4 centred on the alert (time-based events)
      av     — 5s .mkv with audio+video (speaker_audio); falls back to .mp4
               if ffmpeg is unavailable.

    Special cases:
      multiple_people / no_person → image per alert occurrence;
                                    video (.mp4) on termination.

    Usage:
        pw = ProofWriter("reports/proof", fps=20.0, pre_s=2.5, post_s=2.5)
        # in main loop:
        pw.push_frame(frame, time.time())
        # on alert:
        path = pw.save_proof(key, frame, time.time(), audio_monitor=am)
        # on shutdown:
        pw.flush()
    """

    _PROOF_IMAGE = frozenset({"phone", "book", "headphone", "earbud"})
    _PROOF_VIDEO = frozenset({
        "looking_away", "looking_down", "looking_up", "looking_side",
        "face_hidden", "partial_face", "fake_presence",
    })
    _PROOF_AV = frozenset({"speaker_audio"})

    def __init__(
        self,
        proof_dir: str,
        fps: float = 20.0,
        pre_s: float = 2.5,
        post_s: float = 2.5,
        frame_buffer: int = 150,
    ) -> None:
        self._dir    = Path(proof_dir)
        self._fps    = fps
        self._pre_s  = pre_s
        self._post_s = post_s
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=frame_buffer)
        self._pending: list[threading.Thread] = []
        self._ffmpeg = shutil.which("ffmpeg")

    @property
    def has_ffmpeg(self) -> bool:
        return self._ffmpeg is not None

    # ── Public ────────────────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Feed the rolling frame buffer. Call once per main-loop iteration."""
        self._frames.append((timestamp, frame.copy()))

    def save_proof(
        self,
        key: str,
        frame: np.ndarray,
        timestamp: float,
        audio_monitor=None,
        is_termination: bool = False,
    ) -> str | None:
        """
        Save proof for the given event key. Returns the file path immediately
        (even for async video/AV writes, the path is pre-determined).
        Returns None if the key is unknown or unhandled.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H%M%S_%f")[:9]

        if key in self._PROOF_IMAGE:
            return self._save_image(key, frame, ts_str)

        if key in ("multiple_people", "no_person"):
            if is_termination:
                path = str(self._dir / f"{key}_{ts_str}.mp4")
                self._start_video_thread(timestamp, path)
                return path
            else:
                return self._save_image(key, frame, ts_str)

        if key in self._PROOF_VIDEO:
            path = str(self._dir / f"{key}_{ts_str}.mp4")
            self._start_video_thread(timestamp, path)
            return path

        if key in self._PROOF_AV:
            # With ffmpeg: single .mkv; without: .mp4 + companion .wav
            ext  = ".mkv" if self._ffmpeg else ".mp4"
            path = str(self._dir / f"{key}_{ts_str}{ext}")
            self._start_av_thread(timestamp, path, audio_monitor)
            return path

        return None

    def flush(self) -> None:
        """Block until all pending async writes finish."""
        for t in list(self._pending):
            t.join(timeout=10.0)
        self._pending.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_image(self, key: str, frame: np.ndarray, ts_str: str) -> str:
        path = str(self._dir / f"{key}_{ts_str}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _get_pre_frames(self, event_time: float) -> list[tuple[float, np.ndarray]]:
        cutoff = event_time - self._pre_s
        return [(t, f) for t, f in list(self._frames) if t >= cutoff]

    def _start_video_thread(self, event_time: float, path: str) -> None:
        pre = self._get_pre_frames(event_time)
        t = threading.Thread(
            target=self._write_video,
            args=(pre, event_time, path),
            daemon=True,
        )
        t.start()
        self._pending.append(t)

    def _write_video(
        self,
        pre_frames: list[tuple[float, np.ndarray]],
        event_time: float,
        path: str,
    ) -> None:
        time.sleep(self._post_s + 0.3)
        post = [
            (t, f) for t, f in list(self._frames)
            if event_time < t <= event_time + self._post_s
        ]
        all_frames = self._merge(pre_frames, post)
        if not all_frames:
            return
        fps = self._actual_fps(all_frames, self._fps)
        h, w = all_frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for _, frm in all_frames:
            out.write(frm)
        out.release()

    def _start_av_thread(
        self, event_time: float, path: str, audio_monitor
    ) -> None:
        pre = self._get_pre_frames(event_time)
        t = threading.Thread(
            target=self._write_av,
            args=(pre, event_time, path, audio_monitor),
            daemon=True,
        )
        t.start()
        self._pending.append(t)

    def _write_av(
        self,
        pre_frames: list[tuple[float, np.ndarray]],
        event_time: float,
        path: str,
        audio_monitor,
    ) -> None:
        time.sleep(self._post_s + 0.3)
        post = [
            (t, f) for t, f in list(self._frames)
            if event_time < t <= event_time + self._post_s
        ]
        all_frames = self._merge(pre_frames, post)
        if not all_frames:
            return

        fps = self._actual_fps(all_frames, self._fps)
        h, w = all_frames[0][1].shape[:2]
        stem  = path.rsplit(".", 1)[0]
        tmp_v = stem + "_tmpv.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_v, fourcc, fps, (w, h))
        for _, frm in all_frames:
            out.write(frm)
        out.release()

        # Extract audio clip (needed for both merge and companion-file paths)
        audio_raw: bytes = b""
        if audio_monitor is not None:
            t0 = event_time - self._pre_s
            t1 = event_time + self._post_s
            audio_raw = audio_monitor.get_audio_range(t0, t1)

        # ── ffmpeg available → merge into single .mkv ──────────────────────
        if self._ffmpeg and audio_raw:
            tmp_a = stem + "_tmpa.wav"
            self._write_wav(tmp_a, audio_raw,
                            audio_monitor.sample_rate, audio_monitor.channels)
            final = path if path.endswith(".mkv") else stem + ".mkv"
            subprocess.run(
                [self._ffmpeg, "-y",
                 "-i", tmp_v,
                 "-i", tmp_a,
                 "-c:v", "copy", "-c:a", "aac",
                 final],
                capture_output=True,
            )
            try:
                os.unlink(tmp_a)
                os.unlink(tmp_v)
            except OSError:
                pass
            return

        # ── No ffmpeg → video .mp4 + companion .wav ────────────────────────
        final_v = stem + ".mp4"
        try:
            os.replace(tmp_v, final_v)
        except OSError:
            pass

        if audio_raw and audio_monitor is not None:
            self._write_wav(
                stem + ".wav", audio_raw,
                audio_monitor.sample_rate, audio_monitor.channels,
            )

    @staticmethod
    def _actual_fps(
        frames: list[tuple[float, np.ndarray]], fallback: float
    ) -> float:
        """Derive fps from the real wall-clock span of the captured frames."""
        if len(frames) < 2:
            return fallback
        span = frames[-1][0] - frames[0][0]
        if span < 0.1:
            return fallback
        return (len(frames) - 1) / span

    @staticmethod
    def _merge(
        pre:  list[tuple[float, np.ndarray]],
        post: list[tuple[float, np.ndarray]],
    ) -> list[tuple[float, np.ndarray]]:
        combined: dict[float, np.ndarray] = {t: f for t, f in pre}
        combined.update({t: f for t, f in post})
        return sorted(combined.items(), key=lambda x: x[0])

    @staticmethod
    def _write_wav(path: str, raw: bytes, sample_rate: int, channels: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw)
