import cv2
import json
import os
import time
from datetime import datetime

import numpy as np

from config import *
from utils import AlertManager, draw_alerts, draw_audio_status, draw_detections, ProofWriter
from detectors import HeadPoseDetector, LipDetector, ObjectDetector, merge_by_class
from core import (
    AlertEngine,
    AudioMonitor,
    HeadTracker,
    LivenessDetector,
    ObjectTemporalTracker,
    RiskEngine,
    ExamState,
    SpeakerAudioDetector,
)

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

    # Expand panel height if any continuous-timer debug lines are needed
    multi_dur = risk.continuous_duration("multiple_people")
    no_dur    = risk.continuous_duration("no_person")
    extra_lines = (1 if multi_dur > 0 else 0) + (1 if no_dur > 0 else 0)

    panel_w, panel_h = 220, 82 + extra_lines * 18
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

    # Continuous-timer debug lines (shown only when active)
    debug_y = y1 + 76 + 18
    if multi_dur > 0:
        bar_frac = min(multi_dur / 20.0, 1.0)
        timer_color = (0, 0, 255) if multi_dur >= 15 else (0, 165, 255)
        cv2.putText(frame, f"MultiPpl: {multi_dur:.1f}s / 20s",
                    (x1 + 6, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, timer_color, 1)
        bar_x2 = bx1 + int(bw * bar_frac)
        cv2.rectangle(frame, (bx1, debug_y + 3), (bx2, debug_y + 7), (55, 55, 55), -1)
        if bar_x2 > bx1:
            cv2.rectangle(frame, (bx1, debug_y + 3), (bar_x2, debug_y + 7), timer_color, -1)
        debug_y += 18
    if no_dur > 0:
        bar_frac = min(no_dur / 20.0, 1.0)
        timer_color = (0, 0, 255) if no_dur >= 15 else (0, 165, 255)
        cv2.putText(frame, f"NoPerson: {no_dur:.1f}s / 20s",
                    (x1 + 6, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, timer_color, 1)
        bar_x2 = bx1 + int(bw * bar_frac)
        cv2.rectangle(frame, (bx1, debug_y + 3), (bx2, debug_y + 7), (55, 55, 55), -1)
        if bar_x2 > bx1:
            cv2.rectangle(frame, (bx1, debug_y + 3), (bar_x2, debug_y + 7), timer_color, -1)

    if info["terminated"]:
        cv2.rectangle(frame, (0, h // 2 - 32), (w, h // 2 + 32), (0, 0, 180), -1)
        cv2.putText(frame, "EXAM TERMINATED",
                    (w // 2 - 165, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


# ── Partial face banner ───────────────────────────────────────────────────────

def _draw_partial_face_banner(frame: np.ndarray) -> None:
    """Bold bottom-of-frame banner when candidate is too far from camera."""
    h, w = frame.shape[:2]
    banner_h = 70
    y1 = h - banner_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (w, h), (0, 120, 255), -1)   # deep orange
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    msg  = "MOVE CLOSER TO CAMERA"
    sz   = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
    cv2.putText(frame, msg,
                ((w - sz[0]) // 2, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)

    sub  = "Face too small — earphone detection may be impaired"
    sz2  = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, sub,
                ((w - sz2[0]) // 2, y1 + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ── Report ────────────────────────────────────────────────────────────────────

def _save_report(session_start: float, alert_log: list, warning_log: list,
                 risk_summary: dict, session_dir: str) -> None:
    if not SAVE_REPORT:
        return
    os.makedirs(session_dir, exist_ok=True)

    end_time = time.time()
    alert_summary: dict[str, int] = {}
    for entry in alert_log:
        k = entry["message"].split("(")[0].strip()
        alert_summary[k] = alert_summary.get(k, 0) + 1

    warn_summary: dict[str, int] = {}
    for entry in warning_log:
        k = entry["message"]
        warn_summary[k] = warn_summary.get(k, 0) + 1

    report = {
        "session_start"   : datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M:%S"),
        "session_end"     : datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s"      : round(end_time - session_start, 1),
        "total_api_alerts": len(alert_log),
        "total_warnings"  : len(warning_log),
        "alert_summary"   : alert_summary,
        "warning_summary" : warn_summary,
        "alert_log"       : alert_log,
        "warning_log"     : warning_log,
        "risk"            : risk_summary,
    }

    path = os.path.join(session_dir, "report.json")
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
    session_clock = time.monotonic()

    alert_log:   list[dict] = []
    warning_log: list[dict] = []

    # ── States dict ──────────────────────────────────────────────────────────
    states = {
        "phone"          : {"active": False, "last_alert": 0},
        "multiple_people": {"active": False, "last_alert": 0},
        "no_person"      : {"active": False, "last_alert": 0},
        "book"           : {"active": False, "last_alert": 0},
        "headphone"      : {"active": False, "last_alert": 0},
        "earbud"         : {"active": False, "last_alert": 0},
        "speaker_audio"  : {"active": False, "last_alert": 0},
        "looking_away"   : {"active": False, "last_alert": 0, "start_time": None},
        "looking_down"   : {"active": False, "last_alert": 0, "start_time": None},
        "looking_up"     : {"active": False, "last_alert": 0, "start_time": None},
        "looking_side"   : {"active": False, "last_alert": 0, "start_time": None},
        "face_hidden"    : {"active": False, "last_alert": 0, "start_time": None},
        "partial_face"   : {"active": False, "last_alert": 0, "start_time": None},
        "fake_presence"  : {"active": False, "last_alert": 0, "start_time": None},
    }

    # ── Alert manager ─────────────────────────────────────────────────────────
    alert_manager = AlertManager()

    def _on_api_alert(message: str) -> None:
        elapsed = time.time() - session_start
        m, s    = divmod(int(elapsed), 60)
        alert_log.append({
            "time"     : f"{m:02d}:{s:02d}",
            "elapsed_s": round(elapsed, 1),
            "message"  : message,
        })

    def _on_warn_notice(message: str) -> None:
        elapsed = time.time() - session_start
        m, s    = divmod(int(elapsed), 60)
        warning_log.append({
            "time"     : f"{m:02d}:{s:02d}",
            "elapsed_s": round(elapsed, 1),
            "message"  : message,
        })

    alert_manager.on_alert = lambda msg: _on_api_alert(msg)
    alert_manager.on_warn  = lambda msg: _on_warn_notice(msg)

    # ── Session folder ────────────────────────────────────────────────────────
    session_id  = datetime.fromtimestamp(session_start).strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(REPORT_DIR, session_id)
    proof_dir   = os.path.join(session_dir, "proof")
    os.makedirs(session_dir, exist_ok=True)

    # ── Components ────────────────────────────────────────────────────────────
    detector     = ObjectDetector()
    head_detector = HeadPoseDetector(DEBUG)
    lip_detector  = LipDetector()
    obj_tracker  = ObjectTemporalTracker(
        window=OBJECT_WINDOW, min_votes=OBJECT_MIN_VOTES,
        per_key_min_votes={
            "phone" : PHONE_MIN_VOTES,
            "book"  : BOOK_MIN_VOTES,
            "earbud": EARBUD_MIN_VOTES,
        },
    )
    alert_engine = AlertEngine()
    head_tracker = HeadTracker(states, LOOKING_AWAY_THRESHOLD, debug=DEBUG and DEBUG_MEDIAPIPE)
    liveness     = LivenessDetector(FAKE_WINDOW, SAMPLE_INTERVAL, MIN_VARIANCE,
                                    NO_BLINK_TIMEOUT, LIVENESS_WEIGHTS)
    audio_monitor = AudioMonitor(
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        chunk_samples=AUDIO_CHUNK_SAMPLES,
        speech_threshold=AUDIO_SPEECH_THRESH,
    )
    speaker_audio = SpeakerAudioDetector(hold_s=SPEAKER_HOLD_S)
    risk          = RiskEngine(session_duration_s=RISK_SESSION_DURATION_S,
                               flicker_grace_s=TIMER_FLICKER_GRACE_S)
    proof_writer  = ProofWriter(
        proof_dir,
        fps=PROOF_VIDEO_FPS,
        pre_s=PROOF_PRE_S,
        post_s=PROOF_POST_S,
    )
    audio_monitor.start()

    # ── Event handler: alert engine + optional proof capture ─────────────────
    _termination_proved = [False]   # mutable flag — proof saved only once

    def _handle_event(rev, frame: np.ndarray, now: float) -> None:
        alert_engine.handle(rev, alert_manager)

        # Always attach score_added to the most recent log entry
        if rev.risk_added > 0 and alert_log:
            alert_log[-1]["score_added"] = round(rev.risk_added, 2)

        if not SAVE_PROOF:
            return

        # Termination proof — only once, for the key that caused it
        if rev.terminated:
            if not _termination_proved[0]:
                _termination_proved[0] = True
                path = proof_writer.save_proof(
                    rev.key, frame, now, is_termination=True,
                )
                if path and alert_log:
                    alert_log[-1]["proof"] = path
            return

        # Regular proof on scoring alerts
        if rev.risk_added <= 0:
            return

        needs_audio = rev.key in ProofWriter._PROOF_AV
        path = proof_writer.save_proof(
            rev.key, frame, now,
            audio_monitor=audio_monitor if needs_audio else None,
        )
        if path and alert_log:
            alert_log[-1]["proof"] = path

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

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
        ts  = time.monotonic() - session_clock

        # Feed proof frame buffer
        if SAVE_PROOF:
            proof_writer.push_frame(frame, now)

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
        ) = head_detector.detect(
            frame,
            draw         = DEBUG,
            show_gaze    = DEBUG and DEBUG_MEDIAPIPE and DETECT_LOOKING_SIDE,
            show_pose    = DEBUG and DEBUG_MEDIAPIPE and (DETECT_LOOKING_AWAY or DETECT_LOOKING_DOWN or DETECT_LOOKING_UP),
            show_liveness= DEBUG and DEBUG_MEDIAPIPE and DETECT_FAKE_PRESENCE,
        )

        # ── Liveness ──────────────────────────────────────────────────────────
        liveness.update(yaw, pitch, gaze, blinked)
        fake, _ = liveness.is_fake()

        # ── Lip / audio detection ─────────────────────────────────────────────
        _draw_lip = DEBUG and DEBUG_AUDIO and DETECT_SPEAKER_AUDIO
        lip_state = lip_detector.process(frame, ts, draw=_draw_lip)
        speech_active = audio_monitor.speech_active() if DETECT_SPEAKER_AUDIO else False
        speaker_flagged = (
            speaker_audio.update(
                speech_active=speech_active,
                lip_speaking=lip_state.is_speaking,
                face_detected=lip_state.face_detected,
                timestamp=ts,
            )
            if DETECT_SPEAKER_AUDIO else False
        )

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
        face_hidden_cond = people_count > 0 and not (yaw or pitch or gaze)
        no_person_cond   = people_count == 0 and not (yaw or pitch or gaze)

        head_conditions: dict[str, tuple[bool, bool]] = {
            "looking_away" : (looking_away,                  DETECT_LOOKING_AWAY),
            "looking_down" : (looking_down,                  DETECT_LOOKING_DOWN),
            "looking_up"   : (looking_up,                    DETECT_LOOKING_UP),
            "looking_side" : (looking_left or looking_right, DETECT_LOOKING_SIDE),
            "face_hidden"  : (face_hidden_cond,              DETECT_FACE_HIDDEN),
            "partial_face" : (partial_face,                  DETECT_PARTIAL_FACE),
            "fake_presence": (fake and not no_person_cond,   DETECT_FAKE_PRESENCE),
        }

        partial_face_triggered = False
        for key, (cond, enabled) in head_conditions.items():
            if not enabled:
                continue

            t = GAZE_THRESHOLD if key == "looking_side" else None
            triggered = head_tracker.process(frame, key, cond, threshold=t)

            if key == "partial_face":
                partial_face_triggered = triggered

            dur = _get_duration(states, key) if triggered else 0.0
            rev = risk.process_event(key, triggered, confidence=1.0, duration=dur)
            _handle_event(rev, frame, now)

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
            _handle_event(rev, frame, now)

        # ── Multiple people ───────────────────────────────────────────────────
        if DETECT_MULTIPLE_PEOPLE:
            rev = risk.process_event("multiple_people", people_count > 1)
            _handle_event(rev, frame, now)

        # ── No person ─────────────────────────────────────────────────────────
        rev = risk.process_event("no_person", no_person_cond)
        _handle_event(rev, frame, now)

        # ── Speaker audio ─────────────────────────────────────────────────────
        if DETECT_SPEAKER_AUDIO:
            rev = risk.process_event("speaker_audio", speaker_flagged)
            _handle_event(rev, frame, now)

        # ── Drawing ───────────────────────────────────────────────────────────
        if DEBUG and DEBUG_BBOX:
            draw_detections(frame, detections)
        if DRAW_ALERTS:
            draw_alerts(
                frame,
                alert_manager.get_active_warnings(),
                alert_manager.get_active_alerts(),
            )
        if DEBUG and DEBUG_AUDIO and DETECT_SPEAKER_AUDIO:
            draw_audio_status(frame, speech_active)

        if DRAW_RISK_OVERLAY:
            _draw_risk_overlay(frame, risk)

        if partial_face_triggered:
            _draw_partial_face_banner(frame)

        cv2.imshow("AI Proctor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    lip_detector.close()
    audio_monitor.stop()
    proof_writer.flush()

    _save_report(session_start, alert_log, warning_log, risk.get_summary(), session_dir)


if __name__ == "__main__":
    main()
