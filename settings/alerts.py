# ═══════════════════════════════════════════════════════════════════════════════
#  AI Proctor — Alert & Warning Settings
#
#  RULES (must be respected when editing):
#    api_cooldown  == score_cooldown   every score event directly fires an alert
#    warn_cooldown <= score_cooldown   warning shows at least as often as scoring
#
#  Grace period is enforced in the RiskEngine (occ=1 arms cooldown, no score).
#  AlertEngine rule: risk_added == 0 → WARNING,  risk_added > 0 → ALERT.
# ═══════════════════════════════════════════════════════════════════════════════

# ── On-screen display durations (seconds) ──────────────────────────────────────
WARN_DISPLAY_DURATION  : float = 3.0   # how long a warning banner stays visible
ALERT_DISPLAY_DURATION : float = 5.0   # how long an alert banner stays visible

# ── Warning cooldowns (seconds) ────────────────────────────────────────────────
# Minimum gap between consecutive soft (amber) warnings for the same key.
# Must be <= the corresponding SCORE_COOLDOWNS value.
WARN_COOLDOWNS: dict = {
    "looking_away"   :  3,    # score_cooldown =  5s
    "looking_down"   :  3,
    "looking_up"     :  3,
    "looking_side"   :  3,
    "partial_face"   :  3,
    "face_hidden"    :  3,
    "fake_presence"  :  5,    # score_cooldown = 10s
    "phone"          :  8,    # score_cooldown = 15s
    "multiple_people":  5,    # score_cooldown = 10s
    "no_person"      :  5,
    "book"           : 15,    # score_cooldown = 30s
    "headphone"      : 15,
    "earbud"         : 15,
}

# ── API alert cooldowns (seconds) ─────────────────────────────────────────────
# MUST equal SCORE_COOLDOWNS — every scoring event fires an alert.
# Edit both together when changing timing for a key.
API_COOLDOWNS: dict = {
    "looking_away"   :  5,
    "looking_down"   :  5,
    "looking_up"     :  5,
    "looking_side"   :  5,
    "partial_face"   :  5,
    "face_hidden"    :  5,
    "fake_presence"  : 10,
    "phone"          : 15,
    "multiple_people": 10,
    "no_person"      : 10,
    "book"           : 30,
    "headphone"      : 30,
    "earbud"         : 30,
}

# ── Score preview text (shown inside every warning banner) ─────────────────────
# Describes what score the event will add once it escalates to a scored alert.
# Keep these strings in sync with the score values in scoring.py.
SCORE_PREVIEW: dict = {
    "phone"          : "+25 on 2nd detect",
    "book"           : "+20 per 30s",
    "headphone"      : "+20 on 2nd detect",
    "earbud"         : "+20 on 2nd detect",
    "looking_away"   : "+5 per 5s",
    "looking_down"   : "+5 per 5s",
    "looking_up"     : "+5 per 5s",
    "looking_side"   : "+5 per 5s",
    "face_hidden"    : "+10 at 5s  /  +20 at 10s",
    "partial_face"   : "+2 per 5s",
    "fake_presence"  : "+30 at 10s  /  +60 at 25s",
    "multiple_people": "+20 on 2nd  /  +50 on 3rd",
    "no_person"      : "+25 at 5s  /  +50 at 10s",
}
