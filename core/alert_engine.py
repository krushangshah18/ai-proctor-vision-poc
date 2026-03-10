"""
AlertEngine — decides warn vs API-alert based on RiskEvent from RiskEngine.

Design:
  - warn()  : soft on-screen amber message. No logging. No API call.
              Shown on first occurrence (before risk kicks in).
  - alert() : hard red on-screen + API call + frame snapshot saved as proof.
              Fires only when risk was actually scored AND API cooldown elapsed.

Speaking (video lip detection) is WARN-ONLY — never triggers an API alert
and never interacts with the risk score. Audio events handle scored speaking.
"""

import cv2
import os
import time
from datetime import datetime

from .risk_engine import RiskEvent


# Events that only ever produce a soft warning — no API call, no score impact.
_WARN_ONLY = {"speaking"}

# How many occurrences are "warning-only" before the first API alert is allowed.
# 0  = alert allowed from first occurrence (if risk_added > 0).
# 1  = first occurrence is always a warning, API alert from second onward.
_WARN_FIRST_N: dict[str, int] = {
    "phone"          : 1,
    "multiple_people": 1,
    "headphone"      : 1,
    "earbud"         : 1,
    "fake_presence"  : 1,   # first detection zone is warning (duration < 15s)
    "face_hidden"    : 1,   # first detection zone is warning (duration < 5s)
    "looking_away"   : 1,
    "looking_down"   : 1,
    "looking_up"     : 1,
    "looking_side"   : 1,
    "partial_face"   : 0,
    "book"           : 0,
    "no_person"      : 0,
}

# API-alert cooldowns: minimum seconds between consecutive API alerts for the same key.
# Much longer than risk-scoring cooldowns — prevents spam while score still accumulates.
_API_COOLDOWNS: dict[str, float] = {
    "looking_away"   : 45,
    "looking_down"   : 45,
    "looking_up"     : 45,
    "looking_side"   : 45,
    "partial_face"   : 90,
    "face_hidden"    : 30,
    "fake_presence"  : 60,
    "phone"          : 60,
    "book"           : 120,
    "headphone"      : 180,
    "earbud"         :  60,
    "multiple_people": 30,
    "no_person"      : 30,
}


class AlertEngine:
    """
    Consumes RiskEvent objects (from RiskEngine) and decides whether to
    show a warning or fire an API alert (with frame snapshot as proof).

    No occurrence tracking here — RiskEvent.occurrence_count is authoritative.
    """

    def __init__(self, states: dict, snapshot_dir: str = "reports/snapshots"):
        self.states       = states
        self.snapshot_dir = snapshot_dir

        # API-alert cooldowns: key → earliest wall-time for next API alert
        self._api_cooldown_until: dict[str, float] = {}

        # Whether we have already shown the soft warning in the current active period.
        # Resets on each rising edge (is_new_occurrence).
        self._warned_this_period: dict[str, bool] = {}

        # Set after each handle() call — caller uses these to enrich the log entry.
        self.last_snapshot_path: str | None = None
        self.last_risk_added:    float      = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def handle(self, event: RiskEvent, frame, alert_manager) -> None:
        """
        Main entry point. Call once per frame per detection key.

        event         — RiskEvent returned by RiskEngine.process_event()
        frame         — current BGR frame (for proof snapshot)
        alert_manager — AlertManager instance
        """
        self.last_snapshot_path = None
        self.last_risk_added    = 0.0

        key = event.key

        # ── Terminated state ──────────────────────────────────────────────────
        if event.terminated:
            alert_manager.alert(f"EXAM TERMINATED: {event.termination_reason}")
            self.last_snapshot_path = self._save_snapshot("terminated", frame, time.time())
            return

        # ── Inactive event: nothing to show ───────────────────────────────────
        if not event.active and not event.is_new_occurrence:
            return

        state     = self.states.get(key, {})
        warn_msg  = state.get("warn_message",  key)
        alert_msg = state.get("alert_message", key)
        now       = time.time()

        # ── Reset warn flag on new active period ──────────────────────────────
        if event.is_new_occurrence:
            self._warned_this_period[key] = False

        # ── WARN-ONLY events (video speaking) — never escalate ────────────────
        if key in _WARN_ONLY:
            if not self._warned_this_period.get(key):
                alert_manager.warn(warn_msg)
                self._warned_this_period[key] = True
            return

        # ── First-N occurrences are warnings only ─────────────────────────────
        warn_n = _WARN_FIRST_N.get(key, 0)
        if event.occurrence_count <= warn_n:
            if not self._warned_this_period.get(key):
                alert_manager.warn(warn_msg)
                self._warned_this_period[key] = True
            return

        # ── Duration-zone warning (risk engine returned 0 — still building up) -
        # Show an amber warning while the event is in its warning zone
        # (e.g. face_hidden 2–5s, fake_presence < 15s).
        if event.risk_added == 0:
            if event.is_new_occurrence and not self._warned_this_period.get(key):
                alert_manager.warn(warn_msg)
                self._warned_this_period[key] = True
            return

        # ── Risk was scored → candidate for API alert ─────────────────────────
        api_cd   = _API_COOLDOWNS.get(key, 60)
        api_due  = self._api_cooldown_until.get(key, 0.0)

        if now >= api_due:
            # Fire API alert
            alert_manager.alert(alert_msg)
            self._api_cooldown_until[key] = now + api_cd
            self.last_risk_added    = event.risk_added

            # Save frame snapshot as proof
            self.last_snapshot_path = self._save_snapshot(key, frame, now)
        # else: risk scored but API is in cooldown — score accumulates silently

    def handle_audio_alert(self, label: str, confidence: float,
                            risk_added: float, frame, alert_manager) -> None:
        """
        Convenience method for audio proctoring events.
        These always fire an API alert (caller already enforces audio API cooldown).
        Saves frame snapshot as proof.
        """
        self.last_snapshot_path = None
        self.last_risk_added    = risk_added
        alert_manager.alert(f"[AUDIO] {label} ({confidence:.0%})")
        now = time.time()
        self.last_snapshot_path = self._save_snapshot("audio", frame, now)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_snapshot(self, key: str, frame, now: float) -> str | None:
        """Save frame as JPEG proof. Returns filepath or None on failure."""
        if frame is None:
            return None
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            ts    = datetime.fromtimestamp(now).strftime("%H%M%S_%f")[:9]
            fname = f"{key}_{ts}.jpg"
            path  = os.path.join(self.snapshot_dir, fname)
            cv2.imwrite(path, frame)
            return path
        except Exception:
            return None
