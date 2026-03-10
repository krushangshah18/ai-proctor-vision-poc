import cv2
import json
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr

from config import *
from utils import AlertManager, draw_alerts, draw_detections
from detectors import ObjectDetector, merge_by_class, HeadPoseDetector
from core import AlertEngine, HeadTracker, LivenessDetector, ObjectTemporalTracker
from core.audio_proctoring import ProctorSession, CheatType

_AUDIO_EVENT_LABELS = {
    CheatType.IMPERSONATION:  "ALERT [AUDIO]: Voice impersonation detected",
    CheatType.GHOST_VOICE:    "ALERT [AUDIO]: Ghost voice / pre-recorded playback",
    CheatType.EXTRA_SPEAKER:  "ALERT [AUDIO]: Extra speaker detected",
    CheatType.VOICE_MISMATCH: "ALERT [AUDIO]: Voice mismatch with enrolled student",
}


# ── Report helpers ────────────────────────────────────────────────────────────

def _save_report(session_start: float, alert_log: list, audio_summary: dict) -> None:
    if not SAVE_REPORT:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)

    end_time = time.time()
    duration = end_time - session_start

    # Count occurrences per alert type (strip confidence %)
    summary: dict[str, int] = {}
    for entry in alert_log:
        key = entry["message"].split("(")[0].strip()
        summary[key] = summary.get(key, 0) + 1

    report = {
        "session_start"   : datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M:%S"),
        "session_end"     : datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s"      : round(duration, 1),
        "total_alerts"    : len(alert_log),
        "alert_summary"   : summary,
        "alert_log"       : alert_log,
        "audio"           : audio_summary,
    }

    fname = datetime.fromtimestamp(session_start).strftime("report_%Y%m%d_%H%M%S.json")
    path  = os.path.join(REPORT_DIR, fname)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")

    session_start = time.time()
    alert_log: list[dict] = []

    states = {
        "phone"           : {"active": False, "last_alert": 0, "message": "ALERT: Mobile phone detected"},
        "multiple_people" : {"active": False, "last_alert": 0, "message": "ALERT: Multiple people detected"},
        "no_person"       : {"active": False, "last_alert": 0, "message": "ALERT: No person present"},
        "book"            : {"active": False, "last_alert": 0, "message": "ALERT: Book detected"},
        "headphone"       : {"active": False, "last_alert": 0, "message": "ALERT: Headphone detected"},
        "earbud"          : {"active": False, "last_alert": 0, "message": "ALERT: Earbud detected"},

        "looking_away"    : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is not facing the screen"},
        "looking_down"    : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is looking down for extended duration"},
        "looking_up"      : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is looking up for extended duration"},
        "looking_side"    : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is looking away from the screen (eye gaze detected)"},
        "face_hidden"     : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face not clearly visible (possible obstruction)"},
        "partial_face"    : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face appears too small (candidate may be too far from camera)"},
        "fake_presence"   : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible fake presence detected (no eye blink / low movement)"},
        "speaking"        : {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible Speaking Detected"},
    }

    alert_manager = AlertManager()

    # Wrap add_alert to capture every alert with its timestamp for the report
    _orig_add_alert = alert_manager.add_alert
    def _logged_add_alert(message: str) -> None:
        _orig_add_alert(message)
        elapsed = time.time() - session_start
        m, s = divmod(int(elapsed), 60)
        alert_log.append({"time": f"{m:02d}:{s:02d}", "elapsed_s": round(elapsed, 1), "message": message})
    alert_manager.add_alert = _logged_add_alert

    detector      = ObjectDetector()
    head_pose_detector = HeadPoseDetector(DEBUG)
    object_tracker = ObjectTemporalTracker(window=OBJECT_WINDOW, min_votes=OBJECT_MIN_VOTES)
    alerts  = AlertEngine(alert_manager, states, COOLDOWN_SECONDS, RESET_COOLDOWN_SECONDS)
    tracker = HeadTracker(states, LOOKING_AWAY_THRESHOLD, debug=DEBUG)
    liveness = LivenessDetector(FAKE_WINDOW, SAMPLE_INTERVAL, MIN_VARIANCE, NO_BLINK_TIMEOUT, LIVENESS_WEIGHTS)

    # ── Audio proctoring setup ────────────────────────────────────────────────
    session        = ProctorSession(sample_rate=AUDIO_SR)
    audio_chunk_q  : queue.Queue = queue.Queue(maxsize=100)
    audio_event_q  : queue.Queue = queue.Queue()
    stop_audio     = threading.Event()
    recorded_chunks: list        = []

    _NOISE_PROFILE_CHUNKS = int(0.5 * AUDIO_SR / AUDIO_CHUNK)
    _noise_buf: list            = []
    _noise_profile: np.ndarray | None = None

    def _denoise(chunk: np.ndarray) -> np.ndarray:
        nonlocal _noise_profile
        if _noise_profile is None:
            return chunk
        return nr.reduce_noise(y=chunk, sr=AUDIO_SR,
                               y_noise=_noise_profile,
                               stationary=True, prop_decrease=0.85).astype(np.float32)

    def _audio_sd_callback(indata, frames, time_info, status):  # noqa: ARG001
        try:
            audio_chunk_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _audio_worker():
        nonlocal _noise_profile
        while not stop_audio.is_set():
            try:
                chunk = audio_chunk_q.get(timeout=0.1)
                if _noise_profile is None:
                    _noise_buf.append(chunk)
                    if len(_noise_buf) >= _NOISE_PROFILE_CHUNKS:
                        _noise_profile = np.concatenate(_noise_buf)
                        print(f"[Audio] Noise profile captured ({len(_noise_profile)/AUDIO_SR:.2f}s)")
                    recorded_chunks.append(chunk)
                    continue
                clean = _denoise(chunk)
                recorded_chunks.append(clean)
                for ev in session.push(clean):
                    audio_event_q.put(ev)
            except queue.Empty:
                continue

    if DETECT_AUDIO:
        # Enrollment
        try:
            enroll_audio, enroll_sr = sf.read(ENROLLMENT_WAV, dtype="float32")
            if enroll_audio.ndim > 1:
                enroll_audio = enroll_audio[:, 0]
            enroll_audio = nr.reduce_noise(y=enroll_audio, sr=enroll_sr,
                                           stationary=True, prop_decrease=0.85).astype(np.float32)
            session.enroll(enroll_audio, sr=enroll_sr)
            print(f"[Audio] Enrolled from {ENROLLMENT_WAV} ({len(enroll_audio)/enroll_sr:.1f}s @ {enroll_sr}Hz)")
        except FileNotFoundError:
            print(f"[Audio] {ENROLLMENT_WAV} not found — audio proctoring disabled")
        except ValueError as e:
            print(f"[Audio] Enrollment failed: {e}")

        audio_stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, dtype="float32",
                                      blocksize=AUDIO_CHUNK, callback=_audio_sd_callback)
        audio_stream.start()
        audio_thread = threading.Thread(target=_audio_worker, daemon=True)
        audio_thread.start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Object detection ──────────────────────────────────────────────────
        raw = detector.detect(frame)
        detections = (merge_by_class(raw, ["person", "earbud"], iou_threshold=0.5)
                      if len(raw) > 1 else raw)

        # ── Head / gaze ───────────────────────────────────────────────────────
        (
            looking_away,
            looking_down,
            looking_up,
            looking_left,
            looking_right,
            partial_face,
            yaw,
            pitch,
            gaze,
            _,
            blinked,
            _,
            speaking,
        ) = head_pose_detector.detect(frame, draw=DRAW_HEAD_POSE)

        # ── Liveness ──────────────────────────────────────────────────────────
        liveness.update(yaw, pitch, gaze, blinked)
        fake, _ = liveness.is_fake()

        # ── Audio events ──────────────────────────────────────────────────────
        if DETECT_AUDIO:
            session.update_lip_activity(speaking)
            while not audio_event_q.empty():
                try:
                    ev = audio_event_q.get_nowait()
                    label = _AUDIO_EVENT_LABELS.get(ev.event_type, f"ALERT [AUDIO]: {ev.event_type.name}")
                    alert_manager.add_alert(f"{label} ({ev.confidence:.0%})")
                except queue.Empty:
                    break

        # ── Object flags ──────────────────────────────────────────────────────
        phone = book = headphone = earbud = False
        people_count = 0
        for d in detections:
            cls = d["class"]
            if   cls == "person"    : people_count += 1
            elif cls == "cell_phone": phone      = True
            elif cls == "book"      : book       = True
            elif cls == "headphone" : headphone  = True
            elif cls == "earbud"    : earbud     = True

        # ── Head / gaze conditions (gated by DETECT_* toggles) ───────────────
        face_hidden_condition = not (yaw or pitch or gaze) and people_count == 0

        head_conditions: dict[str, tuple[bool, bool]] = {
            # key: (condition, enabled)
            "looking_away"  : (looking_away,                DETECT_LOOKING_AWAY),
            "looking_down"  : (looking_down,                DETECT_LOOKING_DOWN),
            "looking_up"    : (looking_up,                  DETECT_LOOKING_UP),
            "looking_side"  : (looking_left or looking_right, DETECT_LOOKING_SIDE),
            "face_hidden"   : (face_hidden_condition,       DETECT_FACE_HIDDEN),
            "partial_face"  : (partial_face,                DETECT_PARTIAL_FACE),
            "fake_presence" : (fake,                        DETECT_FAKE_PRESENCE),
            "speaking"      : (speaking,                    DETECT_SPEAKING),
        }

        for key, (cond, enabled) in head_conditions.items():
            if not enabled:
                continue
            t = SPEAKING_THRESHOLD if key == "speaking" else None
            triggered = tracker.process(frame, key, cond, threshold=t)
            alerts.trigger(key, triggered)

        # ── Object stability (gated by DETECT_* toggles) ─────────────────────
        object_flags: dict[str, tuple[bool, bool]] = {
            "phone"    : (phone,     DETECT_PHONE),
            "book"     : (book,      DETECT_BOOK),
            "headphone": (headphone, DETECT_HEADPHONE),
            "earbud"   : (earbud,    DETECT_EARBUD),
        }
        for key, (present, enabled) in object_flags.items():
            if not enabled:
                continue
            stable = object_tracker.update(key, present)
            alerts.trigger(key, stable)

        if DETECT_MULTIPLE_PEOPLE:
            alerts.trigger("multiple_people", people_count > 1)

        # ── Drawing ───────────────────────────────────────────────────────────
        if DEBUG:
            if DRAW_OBJECTS:
                draw_detections(frame, detections)
            if DRAW_ALERTS:
                draw_alerts(frame, alert_manager.get_active_alerts())

        cv2.imshow("AI Proctor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    audio_summary = {}
    if DETECT_AUDIO:
        stop_audio.set()
        audio_stream.stop()
        audio_stream.close()

        if recorded_chunks:
            recording = np.concatenate(recorded_chunks)
            sf.write("session_recording.wav", recording, AUDIO_SR)
            print(f"[Audio] Session saved → session_recording.wav ({len(recording)/AUDIO_SR:.1f}s)")

        audio_summary = session.get_summary()
        # remove non-serialisable CheatEvent objects from summary
        audio_summary.pop("cheat_events", None)

    cap.release()
    cv2.destroyAllWindows()

    _save_report(session_start, alert_log, audio_summary)


if __name__ == "__main__":
    main()
