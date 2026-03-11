"""
AlertEngine — routes RiskEvent to warn or alert.

Rule:
  risk_added == 0  →  warn()   soft amber banner, no log, no score impact
  risk_added  > 0  →  alert()  red banner + API log + snapshot

Grace (occ=1 of occurrence-based events) is enforced in RiskEngine by arming
the score cooldown without adding score — so AlertEngine always sees
risk_added=0 during grace and routes it to warn automatically.
"""

import cv2
import os
import time
from datetime import datetime

import settings.alerts as A
from .risk_engine import RiskEvent


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

        # Warn cooldowns: key → earliest wall-time for next soft warning
        self._warn_cooldown_until: dict[str, float] = {}

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

        # ── Terminated ────────────────────────────────────────────────────────
        if event.terminated:
            alert_manager.alert(f"EXAM TERMINATED: {event.termination_reason}")
            self.last_snapshot_path = self._save_snapshot("terminated", frame, time.time())
            return

        # ── Inactive: nothing to show ─────────────────────────────────────────
        if not event.active and not event.is_new_occurrence:
            return

        state     = self.states.get(key, {})
        warn_msg  = state.get("warn_message",  key)
        alert_msg = state.get("alert_message", key)
        now       = time.time()

        # ── No score added → WARNING ──────────────────────────────────────────
        # Covers: grace period (occ=1, cooldown armed in engine with no score),
        #         score cooldown active, duration gate not yet met.
        if event.risk_added == 0:
            if self._warn_ok(key, now):
                alert_manager.warn(warn_msg)
                self._arm_warn_cooldown(key, now)
            return

        # ── Score added → ALERT ───────────────────────────────────────────────
        # api_cooldown == score_cooldown so this fires on every scoring event.
        api_due = self._api_cooldown_until.get(key, 0.0)
        if now >= api_due:
            score_tag = f"  [+{event.risk_added:.0f} pts]"
            alert_manager.alert(f"{alert_msg}{score_tag}")
            self._api_cooldown_until[key] = now + A.API_COOLDOWNS.get(key, 10)
            self.last_risk_added    = event.risk_added
            self.last_snapshot_path = self._save_snapshot(key, frame, now)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _warn_ok(self, key: str, now: float) -> bool:
        """True if the warn cooldown for this key has elapsed."""
        return now >= self._warn_cooldown_until.get(key, 0.0)

    def _arm_warn_cooldown(self, key: str, now: float) -> None:
        cd = A.WARN_COOLDOWNS.get(key, 5.0)
        self._warn_cooldown_until[key] = now + cd

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
