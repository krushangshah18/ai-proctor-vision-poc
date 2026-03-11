# ════════════════════════════════════════════════════════════════════════════
#  AI Proctor  —  Master Configuration
#  All flags, thresholds, and paths live here.
#  Change a value here and it takes effect everywhere automatically.
# ════════════════════════════════════════════════════════════════════════════

# ── General ──────────────────────────────────────────────────────────────────
DEBUG = True   # show head-pose debug values (yaw/pitch/gaze, MAR, blinks) — see DRAW_HEAD_POSE

# ── Detection Toggles ────────────────────────────────────────────────────────
# Set False to completely skip a detection — no alert, no processing overhead.

# Head / gaze
DETECT_LOOKING_AWAY    = True    # head rotated sideways (yaw)
DETECT_LOOKING_DOWN    = True    # head tilted down (pitch)
DETECT_LOOKING_UP      = True    # head tilted up (pitch)
DETECT_LOOKING_SIDE    = True    # iris gaze left or right
DETECT_FACE_HIDDEN     = True    # no face landmarks + no person detected
DETECT_PARTIAL_FACE    = True    # face too small / far from camera
DETECT_FAKE_PRESENCE   = True    # no blink / low head movement (liveness)

# Objects
DETECT_PHONE           = True
DETECT_BOOK            = True
DETECT_HEADPHONE       = True
DETECT_EARBUD          = True
DETECT_MULTIPLE_PEOPLE = True

# ── Drawing Toggles ──────────────────────────────────────────────────────────
DRAW_HEAD_POSE = True    # iris dots, face-centre lines, nose dot
DRAW_OBJECTS   = True    # YOLO bounding boxes + class labels
DRAW_ALERTS    = True    # alert text overlay on the video frame

# ── Alert Cooldowns ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS       = 3    # seconds before the same alert can fire again
RESET_COOLDOWN_SECONDS = 1    # seconds after condition clears before re-arm

# ── Duration Thresholds (seconds a condition must persist before alerting) ───
LOOKING_AWAY_THRESHOLD = 1.5   # head pose conditions (yaw, pitch, face_hidden, etc.)
GAZE_THRESHOLD         = 1.0   # iris gaze conditions (looking_side) — faster trigger

# ── Head Pose ────────────────────────────────────────────────────────────────
LOOK_AWAY_YAW   = 0.20    # |nose offset / face_width| > this → looking away
LOOK_DOWN_PITCH = 0.13    # nose below face centre (positive down)
LOOK_UP_PITCH   = -0.10   # nose above face centre (negative up)
GAZE_LEFT       = -0.13   # iris left of eye centre  (reduced by 0.02 → more sensitive)
GAZE_RIGHT      =  0.13   # iris right of eye centre (reduced by 0.02 → more sensitive)

# ── Partial Face (too far from camera) ───────────────────────────────────────
# Face pixel size below either threshold → partial_face triggered.
# Tune these to match your camera + typical seating distance.
# At 640×480: width=80px ≈ face fills ~12% of frame width (very far)
#             width=120px ≈ face fills ~19% — better minimum for earphone detection
MIN_FACE_WIDTH  = 80    # pixels — face narrower than this → too far
MIN_FACE_HEIGHT = 95    # pixels — face shorter than this → too far

# ── Blink / EAR ──────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.20   # eye aspect ratio below this → eye considered closed
BLINK_FRAMES  = 2      # frames eye must stay closed to register a blink

# ── Liveness ─────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL  = 0.2
FAKE_WINDOW      = 15.0
MIN_VARIANCE     = 0.001
NO_BLINK_TIMEOUT = 10
LIVENESS_WEIGHTS = {"yaw": 0.45, "gaze": 0.45, "pitch": 0.10}

# ── Object Detection (temporal stability) ────────────────────────────────────
OBJECT_WINDOW    = 15   # rolling frame window
OBJECT_MIN_VOTES = 5    # object must appear in N of last OBJECT_WINDOW frames
PHONE_MIN_VOTES  = 9    # stricter: phone must appear in 9/15 frames (reduces false positives)

# ── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_SESSION_DURATION_S  = 3600  # assumed exam duration in seconds (used for decay interval)
DRAW_RISK_OVERLAY        = True  # show score / state overlay on video frame
TIMER_FLICKER_GRACE_S    = 1.5   # seconds condition can be absent before resetting continuous timers

# ── Session Report ────────────────────────────────────────────────────────────
SAVE_REPORT = True        # write a JSON report when the session ends
REPORT_DIR  = "reports"   # folder where report files are saved
