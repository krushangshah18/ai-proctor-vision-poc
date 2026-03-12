from __future__ import annotations

import threading
from collections import deque

import numpy as np
import pyaudio
import torch
from silero_vad import load_silero_vad


class AudioMonitor:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        chunk_samples: int,
        speech_threshold: float,
        ring_duration_s: float = 30.0,
    ) -> None:
        self.sample_rate    = sample_rate
        self.channels       = channels
        self.chunk_samples  = chunk_samples
        self.speech_threshold = speech_threshold

        # How many chunks to keep in the ring (~ring_duration_s seconds).
        chunks_per_sec  = sample_rate / chunk_samples
        ring_maxlen     = int(chunks_per_sec * ring_duration_s) + 1

        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._speech_detected = False
        # Timestamped ring: (wall_time, raw_pcm_bytes) per chunk
        self._audio_ring: deque[tuple[float, bytes]] = deque(maxlen=ring_maxlen)
        self._thread: threading.Thread | None = None
        self._stream = None
        self._pa     = None
        self._model  = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa is not None:
            self._pa.terminate()

    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_detected

    def get_audio_range(self, t0: float, t1: float) -> bytes:
        """Return concatenated raw PCM bytes for chunks timestamped in [t0, t1]."""
        with self._lock:
            chunks = [data for ts, data in self._audio_ring if t0 <= ts <= t1]
        return b"".join(chunks)

    def _run(self) -> None:
        try:
            import time as _time
            self._model = load_silero_vad()
            self._pa    = pyaudio.PyAudio()
            self._stream = self._pa.open(
                rate=self.sample_rate,
                channels=self.channels,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.chunk_samples,
            )
            while not self._stop_event.is_set():
                data = self._stream.read(self.chunk_samples, exception_on_overflow=False)
                ts   = _time.time()
                pcm  = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                tensor = torch.from_numpy(pcm)
                prob = self._model(tensor, self.sample_rate).item()
                with self._lock:
                    self._speech_detected = prob >= self.speech_threshold
                    self._audio_ring.append((ts, data))
        except Exception as exc:
            self._error = str(exc)


class SpeakerAudioDetector:
    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self._no_lips_since: float | None = None
        self._flagged = False

    def update(
        self,
        speech_active: bool,
        lip_speaking: bool,
        face_detected: bool,
        timestamp: float,
    ) -> bool:
        # Flag if audio is active AND (no face visible OR lips not moving).
        # Covers both: person present but not speaking, and nobody in frame
        # while audio plays from a device.
        desync = speech_active and (not face_detected or not lip_speaking)

        if desync:
            if self._no_lips_since is None:
                self._no_lips_since = timestamp
            elif (not self._flagged) and (timestamp - self._no_lips_since >= self._hold_s):
                self._flagged = True
        else:
            self._no_lips_since = None
            self._flagged = False

        return self._flagged
