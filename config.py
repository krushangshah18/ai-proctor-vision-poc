# ════════════════════════════════════════════════════════════════════════════
#  AI Proctor  —  Master Configuration
#  All flags, thresholds, and paths live here.
#  Change a value here and it takes effect everywhere automatically.
# ════════════════════════════════════════════════════════════════════════════

# ── General ──────────────────────────────────────────────────────────────────
DEBUG = True   # show on-frame debug overlays (yaw/pitch/gaze values, MAR, etc.)

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
DETECT_SPEAKING        = True    # lip-movement speaking detection (video)

# Objects
DETECT_PHONE           = True
DETECT_BOOK            = True
DETECT_HEADPHONE       = True
DETECT_EARBUD          = True
DETECT_MULTIPLE_PEOPLE = True

# Audio proctoring
DETECT_AUDIO           = True    # enables microphone + ProctorSession
                                 # (extra speaker, voice mismatch, ghost voice…)

# ── Drawing Toggles ──────────────────────────────────────────────────────────
DRAW_HEAD_POSE = False    # iris dots, face-centre lines, nose dot
DRAW_OBJECTS   = True    # YOLO bounding boxes + class labels
DRAW_ALERTS    = True    # alert text overlay on the video frame

# ── Alert Cooldowns ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS       = 3    # seconds before the same alert can fire again
RESET_COOLDOWN_SECONDS = 1    # seconds after condition clears before re-arm

# ── Duration Thresholds (seconds a condition must persist before alerting) ───
LOOKING_AWAY_THRESHOLD = 1.5   # used for all head / gaze conditions
SPEAKING_THRESHOLD     = 2.5   # lip movement must persist this long

# ── Head Pose ────────────────────────────────────────────────────────────────
LOOK_AWAY_YAW   = 0.20    # |nose offset / face_width| > this → looking away
LOOK_DOWN_PITCH = 0.13    # nose below face centre (positive down)
LOOK_UP_PITCH   = -0.10   # nose above face centre (negative up)
GAZE_LEFT       = -0.15   # iris left of eye centre
GAZE_RIGHT      =  0.15   # iris right of eye centre

# ── Blink / EAR ──────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.20   # eye aspect ratio below this → eye considered closed
BLINK_FRAMES  = 2      # frames eye must stay closed to register a blink

# ── Speaking / Lip Detection ─────────────────────────────────────────────────
MAR_VAR_THRESHOLD     = 0.0006
MAR_VEL_THRESHOLD     = 0.012
MAR_OSC_THRESHOLD     = 4
MAR_YAWN_THRESHOLD    = 0.45     # mean MAR above this → yawn guard, not speech
SILENCE_VEL_THRESHOLD = 0.005    # recent lip velocity below this → definitely silent
SPEAKING_HOLD_FRAMES  = 4        # frames to bridge inter-word gaps (~133 ms @ 30 fps)

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

# ── Audio Proctoring ─────────────────────────────────────────────────────────
AUDIO_SR       = 16000            # sample rate (Hz)
AUDIO_CHUNK    = 512              # samples per chunk (~32 ms @ 16 kHz)
ENROLLMENT_WAV = "enrollment.wav" # path to reference speaker recording

# ── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_SESSION_DURATION_S = 3600   # assumed exam duration in seconds (used for decay interval)
DRAW_RISK_OVERLAY       = True   # show score / state overlay on video frame

# ── Session Report ────────────────────────────────────────────────────────────
SAVE_REPORT = True        # write a JSON report when the session ends
REPORT_DIR  = "reports"   # folder where report files are saved
