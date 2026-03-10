"""
Risk Scoring Engine — implements proctoring_risk_scoring_design.md

Single source of truth for:
  - Occurrence counts (rising-edge detections)
  - Two-bucket score: _fixed_score (non-decaying) + _decaying_score (decaying)
  - Risk scoring cooldowns (distinct from API-alert cooldowns in AlertEngine)
  - Score decay, combo bonuses, state machine, termination rules
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── State machine ─────────────────────────────────────────────────────────────

class ExamState(Enum):
    NORMAL       = "NORMAL"
    WARNING      = "WARNING"
    HIGH_RISK    = "HIGH_RISK"
    ADMIN_REVIEW = "ADMIN_REVIEW"
    TERMINATED   = "TERMINATED"


# ── RiskEvent — returned by process_event() ───────────────────────────────────

@dataclass
class RiskEvent:
    """Result of processing one event for one frame.

    AlertEngine reads this to decide warn vs API-alert.
    """
    key: str
    active: bool             = False  # current condition state
    risk_added: float        = 0.0    # score added this call (0 = warning zone / cooldown)
    is_new_occurrence: bool  = False  # True on the rising edge frame
    occurrence_count: int    = 0      # total rising-edge detections for this key
    duration: float          = 0.0    # seconds this active period has been running
    terminated: bool         = False
    termination_reason: str  = ""


# ── Constants ─────────────────────────────────────────────────────────────────

# Events whose score contribution never decays (stays in _fixed_score)
_NON_DECAYING = {"tab_switch", "phone", "fake_presence", "multiple_people", "no_person"}

# Minimum confidence for any risk to be applied (confidence weighting: Risk += base × conf)
_MIN_CONF = 0.5

# Risk-scoring cooldowns: how often (seconds) the same event can add to the score.
# These are short (scoring mechanics). API-alert cooldowns are longer and live in AlertEngine.
_SCORE_COOLDOWNS: dict[str, float] = {
    "looking_away"   : 3,
    "looking_down"   : 3,
    "looking_up"     : 3,
    "looking_side"   : 3,
    "partial_face"   : 3,
    "face_hidden"    : 10,
    "book"           : 60,
    "headphone"      : 120,
    "earbud"         :  60,
    "phone"          : 15,
    "multiple_people": 10,
    "no_person"      : 10,
    "fake_presence"  : 15,
}

_GAZE_EVENTS = {"looking_away", "looking_down", "looking_up", "looking_side"}

# State thresholds (score ranges)
_T_WARNING    = 30
_T_HIGH_RISK  = 60
_T_ADMIN      = 100


# ── Engine ────────────────────────────────────────────────────────────────────

class RiskEngine:

    def __init__(self, session_duration_s: float = 3600.0):
        # Two-bucket score
        self._fixed_score:    float = 0.0   # non-decaying
        self._decaying_score: float = 0.0   # decays every interval

        self.state     = ExamState.NORMAL
        self.terminated          = False
        self._termination_reason = ""

        # Decay config
        self._session_duration_s = session_duration_s
        self._decay_interval     = max(60.0, session_duration_s / 20.0)
        self._creation_time      = time.time()
        self._last_decay_time    = time.time()

        # Decay audit log: each tick recorded for the report
        self.decay_log: list[dict] = []

        # Occurrence tracking: rising-edge counts per key (single source of truth)
        self._occurrences: dict[str, int] = {}

        # Previous-frame active state (for edge detection)
        self._prev_active: dict[str, bool] = {}

        # Score-cooldown: key → timestamp after which scoring is allowed again
        self._score_cooldown_until: dict[str, float] = {}

        # Duration tracking: key → wall-time start of current active period
        self._active_since: dict[str, Optional[float]] = {}

        # Gaze aggregation (30-second window)
        self._gaze_timestamps:      list[float] = []
        self._last_gaze_bonus_time: float       = 0.0

        # Continuous-presence timers for termination rules
        self._multi_people_since: Optional[float] = None
        self._no_person_since:    Optional[float] = None

        # Currently-active event set (for combo detection)
        self._active_set: set[str] = set()

        # Combo cooldowns: combo_key → earliest time combo can fire again
        self._combo_cooldowns: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def score(self) -> float:
        return self._fixed_score + self._decaying_score

    def process_event(self, key: str, active: bool,
                      confidence: float = 1.0,
                      duration: float = 0.0) -> RiskEvent:
        """
        Call once per frame per detection key.

        key        — event identifier
        active     — whether the condition is currently true (after duration-gating)
        confidence — detection confidence (YOLO: from model; head/gaze: 1.0)
        duration   — how long this active period has been running (from HeadTracker
                     or internal timer if 0 is passed for object events)

        Returns a RiskEvent the AlertEngine uses to decide warn vs API-alert.
        """
        if self.terminated:
            return RiskEvent(key=key, terminated=True,
                             termination_reason=self._termination_reason)

        now  = time.time()
        prev = self._prev_active.get(key, False)
        self._prev_active[key] = active

        # ── Edge detection ────────────────────────────────────────────────────
        is_new_occurrence = active and not prev

        if is_new_occurrence:
            self._occurrences[key] = self._occurrences.get(key, 0) + 1

        # ── Active-set maintenance (for combos) ───────────────────────────────
        if active:
            self._active_set.add(key)
        else:
            self._active_set.discard(key)

        # ── Duration tracking (internal fallback when caller passes 0) ────────
        if active:
            if self._active_since.get(key) is None:
                self._active_since[key] = now
            # Prefer caller-supplied duration (more accurate for head events)
            eff_duration = duration if duration > 0 else now - self._active_since[key]
        else:
            self._active_since[key] = None
            eff_duration = 0.0

        occ = self._occurrences.get(key, 0)

        # ── Decay (applied every frame, limited by interval) ──────────────────
        self._apply_decay(now)

        # ── Inactive: just update state and return ────────────────────────────
        if not active:
            self._update_state()
            return RiskEvent(key=key, active=False, occurrence_count=occ)

        # ── Termination / special-duration checks ─────────────────────────────
        term_event = self._handle_special(key, active, occ, eff_duration,
                                          confidence, now)
        if term_event is not None:
            return term_event

        # ── Risk scoring ───────────────────────────────────────────────────────
        risk_added = self._score_event(key, confidence, eff_duration, occ, now)

        # ── Gaze aggregation bonus ─────────────────────────────────────────────
        if key in _GAZE_EVENTS and risk_added > 0:
            self._gaze_timestamps.append(now)
            self._gaze_timestamps = [t for t in self._gaze_timestamps if now - t <= 30]
            if (len(self._gaze_timestamps) >= 5
                    and now - self._last_gaze_bonus_time > 30):
                self._add(10.0, decaying=True)
                self._last_gaze_bonus_time = now

        # ── Combo bonuses ──────────────────────────────────────────────────────
        self._check_combos(key, confidence, now)

        self._update_state()

        return RiskEvent(
            key=key,
            active=True,
            risk_added=risk_added,
            is_new_occurrence=is_new_occurrence,
            occurrence_count=occ,
            duration=eff_duration,
        )

    def add_audio_risk(self, base_score: float, confidence: float) -> float:
        """
        Direct score addition for audio proctoring events.
        Audio risk is always decaying.
        Returns the actual score added (0 if confidence too low).
        """
        if confidence < _MIN_CONF:
            return 0.0
        added = base_score * confidence
        self._add(added, decaying=True)
        self._update_state()
        return added

    def occurrence_count(self, key: str) -> int:
        """Rising-edge detection count for a key (authoritative)."""
        return self._occurrences.get(key, 0)

    def get_display(self) -> dict:
        return {
            "score"    : round(self.score, 1),
            "fixed"    : round(self._fixed_score, 1),
            "decaying" : round(self._decaying_score, 1),
            "state"    : self.state.value,
            "terminated": self.terminated,
            "reason"   : self._termination_reason,
        }

    def get_summary(self) -> dict:
        return {
            "final_score"         : round(self.score, 1),
            "fixed_score"         : round(self._fixed_score, 1),
            "decaying_score"      : round(self._decaying_score, 1),
            "final_state"         : self.state.value,
            "occurrences"         : dict(self._occurrences),
            "terminated"          : self.terminated,
            "termination_reason"  : self._termination_reason,
            "decay_ticks"         : len(self.decay_log),
            "decay_log"           : self.decay_log,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add(self, amount: float, decaying: bool) -> float:
        """Add to the appropriate bucket. Returns amount added."""
        amount = max(0.0, amount)
        if decaying:
            self._decaying_score = min(150.0, self._decaying_score + amount)
        else:
            self._fixed_score = min(150.0, self._fixed_score + amount)
        return amount

    def _in_cooldown(self, key: str, now: float) -> bool:
        return now < self._score_cooldown_until.get(key, 0.0)

    def _arm_cooldown(self, key: str, now: float, override_cd: float = 0.0) -> None:
        cd = override_cd if override_cd > 0 else _SCORE_COOLDOWNS.get(key, 3.0)
        self._score_cooldown_until[key] = now + cd

    def _score_event(self, key: str, confidence: float,
                     duration: float, occ: int, now: float) -> float:
        """Apply per-event scoring policy. Returns amount added to score."""
        if confidence < _MIN_CONF:
            return 0.0
        if self._in_cooldown(key, now):
            return 0.0

        decaying = key not in _NON_DECAYING
        added    = 0.0

        # ── Phone ──────────────────────────────────────────────────────────────
        if key == "phone":
            if occ == 1:
                pass             # first occurrence: warning only, score applied on 2nd+
            elif occ == 2:
                added = self._add(25.0 * confidence, decaying=False)
            else:
                added = self._add(50.0 * confidence, decaying=False)

        # ── Headphone / earbud ────────────────────────────────────────────────
        elif key in ("headphone", "earbud"):
            if occ > 1:
                added = self._add(10.0 * confidence, decaying=True)

        # ── Multiple people ───────────────────────────────────────────────────
        elif key == "multiple_people":
            if occ == 2:
                added = self._add(10.0 * confidence, decaying=False)
            elif occ >= 3:
                added = self._add(50.0 * confidence, decaying=False)
            # occ == 1: warning only (handled in AlertEngine)

        # ── Book ──────────────────────────────────────────────────────────────
        elif key == "book":
            added = self._add(10.0 * confidence, decaying=True)

        # ── Fake presence (duration-tiered) ───────────────────────────────────
        elif key == "fake_presence":
            if duration >= 40:
                added = self._add(60.0 * confidence, decaying=False)
            elif duration >= 15:
                added = self._add(30.0 * confidence, decaying=False)
            # < 15s: warning only

        # ── Face hidden (duration-tiered) ─────────────────────────────────────
        elif key == "face_hidden":
            if duration >= 10:
                added = self._add(20.0 * confidence, decaying=True)
            elif duration >= 5:
                added = self._add(10.0 * confidence, decaying=True)
            # < 5s: warning only (2-5s is warning zone)

        # ── Gaze events (Option A: +5 every 3s-cooldown while active) ─────────
        elif key in _GAZE_EVENTS:
            added = self._add(5.0 * confidence, decaying=True)

        # ── Partial face ──────────────────────────────────────────────────────
        elif key == "partial_face":
            if duration >= 3:
                added = self._add(2.0 * confidence, decaying=True)

        # ── Exit fullscreen ───────────────────────────────────────────────────
        elif key == "exit_fullscreen":
            if occ == 1:
                pass  # warning only
            elif duration >= 2:
                added = self._add(5.0 * confidence, decaying=True)

        if added > 0:
            self._arm_cooldown(key, now)

        return added

    def _handle_special(self, key: str, active: bool, occ: int,
                         duration: float, confidence: float,
                         now: float) -> Optional[RiskEvent]:
        """
        Handle continuous-presence termination rules and no-person duration tiers.
        Returns a RiskEvent only if the exam was just terminated.
        Returns None to let normal scoring proceed.
        """
        if key == "multiple_people":
            if self._multi_people_since is None:
                self._multi_people_since = now
            elif now - self._multi_people_since >= 20:
                self._terminate("Multiple people continuously >20s")
                return RiskEvent(key=key, terminated=True,
                                 termination_reason=self._termination_reason)

        elif key == "no_person":
            if self._no_person_since is None:
                self._no_person_since = now
            else:
                dur = now - self._no_person_since
                if dur >= 20:
                    self._terminate("No person detected >20s")
                    return RiskEvent(key=key, terminated=True,
                                     termination_reason=self._termination_reason)
                elif dur >= 10 and not self._in_cooldown("_no_person_50", now):
                    self._add(50.0, decaying=False)
                    self._arm_cooldown("_no_person_50", now, override_cd=15)
                elif dur >= 5 and not self._in_cooldown("_no_person_25", now):
                    self._add(25.0, decaying=False)
                    self._arm_cooldown("_no_person_25", now, override_cd=15)
            return RiskEvent(key=key, active=True, is_new_occurrence=(occ == 1 and duration < 1),
                             occurrence_count=occ, duration=duration,
                             risk_added=0.0)  # scoring handled above, skip normal path

        return None  # proceed to normal _score_event

    def _check_combos(self, key: str, confidence: float, now: float) -> None:
        combos = [
            ({"looking_down", "book"},    15.0, "_combo_down_book"),
            ({"looking_side", "speaking"}, 15.0, "_combo_side_speaking"),
            ({"phone", "looking_down"},   20.0, "_combo_phone_down"),
        ]
        for combo_set, bonus, cd_key in combos:
            if key in combo_set and combo_set.issubset(self._active_set):
                if now >= self._combo_cooldowns.get(cd_key, 0.0):
                    eff_conf = max(confidence, _MIN_CONF)
                    self._add(bonus * eff_conf, decaying=True)
                    self._combo_cooldowns[cd_key] = now + 60.0

    def _apply_decay(self, now: float) -> None:
        """Decay only the decaying bucket. Records every tick in decay_log."""
        if now - self._last_decay_time >= self._decay_interval:
            before  = self._decaying_score
            self._decaying_score = max(0.0, self._decaying_score - 5.0)
            after   = self._decaying_score
            elapsed = now - self._creation_time
            m, s    = divmod(int(elapsed), 60)
            self.decay_log.append({
                "time"             : f"{m:02d}:{s:02d}",
                "elapsed_s"        : round(elapsed, 1),
                "decayed_amount"   : round(before - after, 2),
                "decaying_before"  : round(before, 2),
                "decaying_after"   : round(after, 2),
                "total_score_after": round(self._fixed_score + after, 2),
            })
            self._last_decay_time = now

    def _update_state(self) -> None:
        if self.terminated:
            self.state = ExamState.TERMINATED
            return
        s = self.score
        if s >= _T_ADMIN:
            self.state = ExamState.ADMIN_REVIEW
        elif s >= _T_HIGH_RISK:
            self.state = ExamState.HIGH_RISK
        elif s >= _T_WARNING:
            self.state = ExamState.WARNING
        else:
            self.state = ExamState.NORMAL

    def _terminate(self, reason: str) -> None:
        if not self.terminated:
            self.terminated          = True
            self._termination_reason = reason
            self.state               = ExamState.TERMINATED
