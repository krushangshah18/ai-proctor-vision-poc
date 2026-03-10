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
from core import AlertEngine, HeadTracker, LivenessDetector, ObjectTemporalTracker, RiskEngine, ExamState
from core.audio_proctoring import ProctorSession, CheatType

# ── Audio event registry ─────────────────────────────────────────────────────
# (label, base_risk_score)
# Risk = base_score × confidence  (always goes into the decaying bucket)
# Speaking risk comes ONLY from audio analysis — video lip detection is warn-only.
_AUDIO_EVENTS: dict[CheatType, tuple[str, float]] = {
    CheatType.IMPERSONATION  : ("Voice impersonation detected",      60.0),
    CheatType.GHOST_VOICE    : ("Ghost voice / pre-recorded audio",  50.0),
    CheatType.EXTRA_SPEAKER  : ("Extra speaker detected",            40.0),
    CheatType.VOICE_MISMATCH : ("Voice mismatch with enrolled user", 40.0),
}

# Per-audio-event API cooldowns (seconds between consecutive alerts of same type)
_AUDIO_API_COOLDOWNS: dict[CheatType, float] = {
    CheatType.IMPERSONATION  : 120,
    CheatType.GHOST_VOICE    : 120,
    CheatType.EXTRA_SPEAKER  :  60,
    CheatType.VOICE_MISMATCH :  60,
}

# State overlay colours (BGR)
_STATE_COLORS = {
    ExamState.NORMAL      : (0, 200, 0),
    ExamState.WARNING     : (0, 200, 255),
    ExamState.HIGH_RISK   : (0, 100, 255),
    ExamState.ADMIN_REVIEW: (0, 0, 255),
    ExamState.TERMINATED  : (0, 0, 180),
}


# ── Risk overlay ──────────────────────────────────────────────────────────────

def _draw_risk_overlay(frame: np.ndarray, risk: RiskEngine) -> None:
    """Semi-transparent panel (top-right): score breakdown, state, progress bar."""
    h, w  = frame.shape[:2]
    info  = risk.get_display()
    score = info["score"]
    color = _STATE_COLORS.get(risk.state, (200, 200, 200))

    panel_w, panel_h = 220, 82
    x1 = w - panel_w - 10;  y1 = 10
    x2 = w - 10;             y2 = y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Risk: {score:.0f}  (F:{info['fixed']:.0f} D:{info['decaying']:.0f})",
                (x1 + 6, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    cv2.putText(frame, info["state"],
                (x1 + 6, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    # Progress bar (capped at 100 for display)
    bx1 = x1 + 6;  bx2 = x2 - 6;  by = y1 + 58
    bw  = bx2 - bx1
    filled = int(bw * min(score, 100) / 100)
    cv2.rectangle(frame, (bx1, by - 6), (bx2, by + 3), (55, 55, 55), -1)
    if filled > 0:
        cv2.rectangle(frame, (bx1, by - 6), (bx1 + filled, by + 3), color, -1)

    # State label below bar
    cv2.putText(frame, f"F=fixed  D=decaying",
                (x1 + 6, y1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    if info["terminated"]:
        cv2.rectangle(frame, (0, h // 2 - 32), (w, h // 2 + 32), (0, 0, 180), -1)
        cv2.putText(frame, "EXAM TERMINATED",
                    (w // 2 - 165, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# ── Report ────────────────────────────────────────────────────────────────────

def _save_report(session_start: float, alert_log: list,
                 audio_summary: dict, risk_summary: dict) -> None:
    if not SAVE_REPORT:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)

    end_time = time.time()
    summary: dict[str, int] = {}
    for entry in alert_log:
        k = entry["message"].split("(")[0].strip()
        summary[k] = summary.get(k, 0) + 1

    report = {
        "session_start"    : datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M:%S"),
        "session_end"      : datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s"       : round(end_time - session_start, 1),
        "total_api_alerts" : len(alert_log),
        "alert_summary"    : summary,
        "alert_log"        : alert_log,
        "audio"            : audio_summary,
        "risk"             : risk_summary,
    }

    fname = datetime.fromtimestamp(session_start).strftime("report_%Y%m%d_%H%M%S.json")
    path  = os.path.join(REPORT_DIR, fname)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved → {path}")


# ── Duration helper ───────────────────────────────────────────────────────────

def _get_duration(states: dict, key: str) -> float:
    """Seconds the current active period has been running (from HeadTracker state)."""
    st    = states.get(key, {})
    start = st.get("start_time")
    return (time.time() - start) if start is not None else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")

    session_start = time.time()

    # alert_log only contains HARD API alerts (not soft warnings).
    # Each entry: {time, elapsed_s, message, snapshot_path}
    alert_log: list[dict] = []

    # ── States dict ──────────────────────────────────────────────────────────
    # warn_message  → amber on-screen, not logged
    # alert_message → red on-screen + API call + snapshot saved
    states = {
        "phone": {
            "active": False, "last_alert": 0,
            "warn_message" : "Phone visible in frame",
            "alert_message": "Mobile phone detected",
        },
        "multiple_people": {
            "active": False, "last_alert": 0,
            "warn_message" : "Multiple people in frame",
            "alert_message": "Multiple people detected",
        },
        "no_person": {
            "active": False, "last_alert": 0,
            "warn_message" : "Candidate not visible",
            "alert_message": "No person detected — candidate absent",
        },
        "book": {
            "active": False, "last_alert": 0,
            "warn_message" : "Book visible in frame",
            "alert_message": "Book detected",
        },
        "headphone": {
            "active": False, "last_alert": 0,
            "warn_message" : "Headphones visible",
            "alert_message": "Headphones detected",
        },
        "earbud": {
            "active": False, "last_alert": 0,
            "warn_message" : "Earbuds visible",
            "alert_message": "Earbuds detected",
        },
        "looking_away": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Not facing screen",
            "alert_message": "Candidate not facing screen",
        },
        "looking_down": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Looking down",
            "alert_message": "Candidate looking down (extended)",
        },
        "looking_up": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Looking up",
            "alert_message": "Candidate looking up (extended)",
        },
        "looking_side": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Gaze shifted sideways",
            "alert_message": "Candidate gaze away from screen",
        },
        "face_hidden": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Face not clearly visible",
            "alert_message": "Face obstructed or hidden",
        },
        "partial_face": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "Face partially outside frame",
            "alert_message": "Candidate too far from camera",
        },
        "fake_presence": {
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message" : "No movement detected — please move slightly",
            "alert_message": "Possible fake presence (static image)",
        },
        "speaking": {
            # WARN-ONLY: video lip detection.
            # Audio proctoring events handle scored speaking violations.
            "active": False, "last_alert": 0, "start_time": None,
            "warn_message": "Candidate may be speaking",
        },
    }

    # ── Alert manager ─────────────────────────────────────────────────────────
    alert_manager = AlertManager()

    # on_alert is the "backend API call" — logs to report with proof reference
    def _on_api_alert(message: str, snapshot_path: str | None = None) -> None:
        elapsed = time.time() - session_start
        m, s    = divmod(int(elapsed), 60)
        entry: dict = {
            "time"     : f"{m:02d}:{s:02d}",
            "elapsed_s": round(elapsed, 1),
            "message"  : message,
        }
        if snapshot_path:
            entry["snapshot"] = snapshot_path
        alert_log.append(entry)

    # Wire: AlertManager.alert() → _on_api_alert (no snapshot from here,
    # snapshot path is injected by AlertEngine after saving)
    alert_manager.on_alert = lambda msg: _on_api_alert(msg)

    # ── Components ────────────────────────────────────────────────────────────
    snapshot_dir   = os.path.join(REPORT_DIR, "snapshots")
    detector       = ObjectDetector()
    head_detector  = HeadPoseDetector(DEBUG)
    obj_tracker    = ObjectTemporalTracker(window=OBJECT_WINDOW, min_votes=OBJECT_MIN_VOTES,
                                           per_key_min_votes={"phone": PHONE_MIN_VOTES})
    alert_engine   = AlertEngine(states, snapshot_dir=snapshot_dir)
    head_tracker   = HeadTracker(states, LOOKING_AWAY_THRESHOLD, debug=DEBUG)
    liveness       = LivenessDetector(FAKE_WINDOW, SAMPLE_INTERVAL, MIN_VARIANCE,
                                      NO_BLINK_TIMEOUT, LIVENESS_WEIGHTS)
    risk           = RiskEngine(session_duration_s=RISK_SESSION_DURATION_S)

    def _enrich_last_entry() -> None:
        """Attach score_added and snapshot path to the most recent API alert log entry."""
        if not alert_log:
            return
        entry = alert_log[-1]
        if alert_engine.last_risk_added > 0:
            entry["score_added"] = round(alert_engine.last_risk_added, 2)
        if alert_engine.last_snapshot_path:
            entry["snapshot"] = alert_engine.last_snapshot_path

    # ── Audio proctoring setup ────────────────────────────────────────────────
    proctor_session    = ProctorSession(sample_rate=AUDIO_SR)
    audio_chunk_q: queue.Queue = queue.Queue(maxsize=100)
    audio_event_q: queue.Queue = queue.Queue()
    stop_audio         = threading.Event()
    recorded_chunks: list      = []
    _audio_last_api: dict[CheatType, float] = {}

    _NOISE_PROFILE_CHUNKS = int(0.5 * AUDIO_SR / AUDIO_CHUNK)
    _noise_buf: list          = []
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
                        print(f"[Audio] Noise profile captured "
                              f"({len(_noise_profile)/AUDIO_SR:.2f}s)")
                    recorded_chunks.append(chunk)
                    continue
                clean = _denoise(chunk)
                recorded_chunks.append(clean)
                for ev in proctor_session.push(clean):
                    audio_event_q.put(ev)
            except queue.Empty:
                continue

    if DETECT_AUDIO:
        try:
            enroll_audio, enroll_sr = sf.read(ENROLLMENT_WAV, dtype="float32")
            if enroll_audio.ndim > 1:
                enroll_audio = enroll_audio[:, 0]
            enroll_audio = nr.reduce_noise(y=enroll_audio, sr=enroll_sr,
                                           stationary=True,
                                           prop_decrease=0.85).astype(np.float32)
            proctor_session.enroll(enroll_audio, sr=enroll_sr)
            print(f"[Audio] Enrolled from {ENROLLMENT_WAV} "
                  f"({len(enroll_audio)/enroll_sr:.1f}s @ {enroll_sr}Hz)")
        except FileNotFoundError:
            print(f"[Audio] {ENROLLMENT_WAV} not found — audio proctoring disabled")
        except ValueError as e:
            print(f"[Audio] Enrollment failed: {e}")

        audio_stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, dtype="float32",
                                      blocksize=AUDIO_CHUNK, callback=_audio_sd_callback)
        audio_stream.start()
        threading.Thread(target=_audio_worker, daemon=True).start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Skip glitched / corrupted frames (near-black or near-uniform)
        # These cause false detections (e.g. spurious multiple_people).
        if frame.mean() < 5 or frame.std() < 8:
            continue

        # Show terminated banner and wait for quit
        if risk.terminated:
            _draw_risk_overlay(frame, risk)
            cv2.imshow("AI Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        now = time.time()

        # ── Object detection ──────────────────────────────────────────────────
        raw        = detector.detect(frame)
        detections = (merge_by_class(raw, ["person", "earbud"], iou_threshold=0.5)
                      if len(raw) > 1 else raw)

        # ── Head / gaze detection ─────────────────────────────────────────────
        (
            looking_away, looking_down, looking_up,
            looking_left, looking_right,
            partial_face,
            yaw, pitch, gaze,
            _, blinked, _,
            speaking,
        ) = head_detector.detect(frame, draw=DRAW_HEAD_POSE)

        # ── Liveness ──────────────────────────────────────────────────────────
        liveness.update(yaw, pitch, gaze, blinked)
        fake, _ = liveness.is_fake()

        # ── Object flags ──────────────────────────────────────────────────────
        phone = book = headphone = earbud = False
        phone_conf = book_conf = hp_conf = eb_conf = 1.0
        people_count = 0
        for d in detections:
            cls  = d["class"]
            conf = d.get("confidence", 1.0)
            if   cls == "person"    : people_count += 1
            elif cls == "cell_phone": phone     = True; phone_conf = conf
            elif cls == "book"      : book      = True; book_conf  = conf
            elif cls == "headphone" : headphone = True; hp_conf    = conf
            elif cls == "earbud"    : earbud    = True; eb_conf    = conf

        # ── Head / gaze: duration-gate → risk → alert ─────────────────────────
        face_hidden_cond = not (yaw or pitch or gaze) and people_count == 0

        head_conditions: dict[str, tuple[bool, bool]] = {
            "looking_away" : (looking_away,                  DETECT_LOOKING_AWAY),
            "looking_down" : (looking_down,                  DETECT_LOOKING_DOWN),
            "looking_up"   : (looking_up,                    DETECT_LOOKING_UP),
            "looking_side" : (looking_left or looking_right, DETECT_LOOKING_SIDE),
            "face_hidden"  : (face_hidden_cond,              DETECT_FACE_HIDDEN),
            "partial_face" : (partial_face,                  DETECT_PARTIAL_FACE),
            "fake_presence": (fake,                          DETECT_FAKE_PRESENCE),
            "speaking"     : (speaking,                      DETECT_SPEAKING),
        }

        for key, (cond, enabled) in head_conditions.items():
            if not enabled:
                continue

            # Duration gate: HeadTracker returns True only after threshold
            t        = SPEAKING_THRESHOLD if key == "speaking" else None
            triggered = head_tracker.process(frame, key, cond, threshold=t)

            # Duration of current active period (from HeadTracker's start_time)
            dur = _get_duration(states, key) if triggered else 0.0

            # Risk engine: score update + occurrence tracking (single source of truth)
            # speaking is passed but RiskEngine has no policy for it — AlertEngine
            # intercepts it as WARN_ONLY before any scoring happens.
            rev = risk.process_event(key, triggered, confidence=1.0, duration=dur)

            # Alert engine: decide warn vs API-alert, save snapshot
            alert_engine.handle(rev, frame, alert_manager)
            _enrich_last_entry()

        # ── Object stability: risk → alert ────────────────────────────────────
        object_flags: dict[str, tuple[bool, bool, float]] = {
            "phone"    : (phone,     DETECT_PHONE,     phone_conf),
            "book"     : (book,      DETECT_BOOK,      book_conf),
            "headphone": (headphone, DETECT_HEADPHONE, hp_conf),
            "earbud"   : (earbud,    DETECT_EARBUD,    eb_conf),
        }
        for key, (present, enabled, conf) in object_flags.items():
            if not enabled:
                continue
            stable = obj_tracker.update(key, present)
            rev    = risk.process_event(key, stable, confidence=conf)
            alert_engine.handle(rev, frame, alert_manager)
            _enrich_last_entry()

        # ── Multiple people ───────────────────────────────────────────────────
        if DETECT_MULTIPLE_PEOPLE:
            rev = risk.process_event("multiple_people", people_count > 1)
            alert_engine.handle(rev, frame, alert_manager)
            _enrich_last_entry()

        # ── No person ─────────────────────────────────────────────────────────
        no_person_cond = people_count == 0 and not (yaw or pitch or gaze)
        rev = risk.process_event("no_person", no_person_cond)
        alert_engine.handle(rev, frame, alert_manager)
        if alert_engine.last_snapshot_path and alert_log:
            alert_log[-1]["snapshot"] = alert_engine.last_snapshot_path

        # ── Audio events: score (decaying) + API alert + snapshot ─────────────
        if DETECT_AUDIO:
            proctor_session.update_lip_activity(speaking)

            while not audio_event_q.empty():
                try:
                    ev = audio_event_q.get_nowait()
                except queue.Empty:
                    break

                label, base_score = _AUDIO_EVENTS.get(
                    ev.event_type, (f"Audio: {ev.event_type.name}", 30.0)
                )
                api_cd   = _AUDIO_API_COOLDOWNS.get(ev.event_type, 60)
                last_api = _audio_last_api.get(ev.event_type, 0.0)

                if (now - last_api) >= api_cd:
                    # Add to decaying risk bucket first
                    added = risk.add_audio_risk(base_score, ev.confidence)
                    if added > 0:
                        # Fire API alert + save proof snapshot
                        alert_engine.handle_audio_alert(
                            label, ev.confidence, added, frame, alert_manager
                        )
                        _audio_last_api[ev.event_type] = now
                        # Attach snapshot to log entry
                        if alert_engine.last_snapshot_path and alert_log:
                            alert_log[-1]["snapshot"] = alert_engine.last_snapshot_path

        # ── Drawing ───────────────────────────────────────────────────────────
        if DEBUG:
            if DRAW_OBJECTS:
                draw_detections(frame, detections)
            if DRAW_ALERTS:
                draw_alerts(
                    frame,
                    alert_manager.get_active_warnings(),
                    alert_manager.get_active_alerts(),
                )

        if DRAW_RISK_OVERLAY:
            _draw_risk_overlay(frame, risk)

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
            print(f"[Audio] Session saved → session_recording.wav "
                  f"({len(recording)/AUDIO_SR:.1f}s)")

        audio_summary = proctor_session.get_summary()
        audio_summary.pop("cheat_events", None)

    cap.release()
    cv2.destroyAllWindows()

    _save_report(session_start, alert_log, audio_summary, risk.get_summary())


if __name__ == "__main__":
    main()
