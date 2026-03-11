import cv2
import json
import os
import time
from datetime import datetime

import numpy as np

from config import *
from utils import AlertManager, draw_alerts, draw_detections
from detectors import ObjectDetector, merge_by_class, HeadPoseDetector
from core import AlertEngine, HeadTracker, LivenessDetector, ObjectTemporalTracker, RiskEngine, ExamState

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
        # Mini progress bar toward 20s termination threshold
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

    # Main instruction — large, centred
    msg  = "MOVE CLOSER TO CAMERA"
    sz   = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 3)[0]
    cv2.putText(frame, msg,
                ((w - sz[0]) // 2, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)

    # Sub-text — smaller
    sub  = "Face too small — earphone detection may be impaired"
    sz2  = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, sub,
                ((w - sz2[0]) // 2, y1 + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


# ── Report ────────────────────────────────────────────────────────────────────

def _save_report(session_start: float, alert_log: list, warning_log: list,
                 risk_summary: dict) -> None:
    if not SAVE_REPORT:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)

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
        "session_start"    : datetime.fromtimestamp(session_start).strftime("%Y-%m-%d %H:%M:%S"),
        "session_end"      : datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s"       : round(end_time - session_start, 1),
        "total_api_alerts" : len(alert_log),
        "total_warnings"   : len(warning_log),
        "alert_summary"    : alert_summary,
        "warning_summary"  : warn_summary,
        "alert_log"        : alert_log,
        "warning_log"      : warning_log,
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

    # alert_log  — hard API alerts only (red, logged, snapshot saved)
    # warning_log — soft warnings (amber, first-occurrence notices)
    alert_log:   list[dict] = []
    warning_log: list[dict] = []

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

    def _on_warn_notice(message: str) -> None:
        elapsed = time.time() - session_start
        m, s    = divmod(int(elapsed), 60)
        warning_log.append({
            "time"     : f"{m:02d}:{s:02d}",
            "elapsed_s": round(elapsed, 1),
            "message"  : message,
        })

    # Wire: AlertManager callbacks → report logs
    alert_manager.on_alert = lambda msg: _on_api_alert(msg)
    alert_manager.on_warn  = lambda msg: _on_warn_notice(msg)

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
    risk           = RiskEngine(session_duration_s=RISK_SESSION_DURATION_S,
                                flicker_grace_s=TIMER_FLICKER_GRACE_S)

    def _enrich_last_entry() -> None:
        """Attach score_added and snapshot path to the most recent API alert log entry."""
        if not alert_log:
            return
        entry = alert_log[-1]
        if alert_engine.last_risk_added > 0:
            entry["score_added"] = round(alert_engine.last_risk_added, 2)
        if alert_engine.last_snapshot_path:
            entry["snapshot"] = alert_engine.last_snapshot_path

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
        # face_hidden: person present (YOLO detects) but face landmarks absent (obscured)
        # no_person: nobody detected by either YOLO or MediaPipe
        face_hidden_cond = people_count > 0 and not (yaw or pitch or gaze)
        no_person_cond   = people_count == 0 and not (yaw or pitch or gaze)

        head_conditions: dict[str, tuple[bool, bool]] = {
            "looking_away" : (looking_away,                  DETECT_LOOKING_AWAY),
            "looking_down" : (looking_down,                  DETECT_LOOKING_DOWN),
            "looking_up"   : (looking_up,                    DETECT_LOOKING_UP),
            "looking_side" : (looking_left or looking_right, DETECT_LOOKING_SIDE),
            "face_hidden"  : (face_hidden_cond,              DETECT_FACE_HIDDEN),
            "partial_face" : (partial_face,                  DETECT_PARTIAL_FACE),
            # Suppress fake_presence when no person is detected — liveness gets
            # all-zero inputs (no face) which falsely looks like "no movement".
            "fake_presence": (fake and not no_person_cond,   DETECT_FAKE_PRESENCE),
        }

        partial_face_triggered = False
        for key, (cond, enabled) in head_conditions.items():
            if not enabled:
                continue

            # Duration gate: HeadTracker returns True only after threshold
            if key == "looking_side":
                t = GAZE_THRESHOLD
            else:
                t = None
            triggered = head_tracker.process(frame, key, cond, threshold=t)

            if key == "partial_face":
                partial_face_triggered = triggered

            # Duration of current active period (from HeadTracker's start_time)
            dur = _get_duration(states, key) if triggered else 0.0

            # Risk engine: score update + occurrence tracking (single source of truth)
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
        rev = risk.process_event("no_person", no_person_cond)
        alert_engine.handle(rev, frame, alert_manager)
        if alert_engine.last_snapshot_path and alert_log:
            alert_log[-1]["snapshot"] = alert_engine.last_snapshot_path

        # ── Drawing ───────────────────────────────────────────────────────────
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

        # Partial face banner — drawn last so it sits on top of everything
        if partial_face_triggered:
            _draw_partial_face_banner(frame)

        cv2.imshow("AI Proctor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Shutdown ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    _save_report(session_start, alert_log, warning_log, risk.get_summary())


if __name__ == "__main__":
    main()
