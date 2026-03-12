# ════════════════════════════════════════════════════════════════════════════
#  AI Proctor  —  Master Configuration
#  All flags, thresholds, and paths live here.
#  Change a value here and it takes effect everywhere automatically.
# ════════════════════════════════════════════════════════════════════════════

# ── General ──────────────────────────────────────────────────────────────────
# DEBUG = False → clean exam view: only alerts/warnings + score overlay visible.
# DEBUG = True  → enables sub-gates below for detailed diagnostic overlays.
DEBUG = True

# Debug sub-gates (only active when DEBUG = True)
DEBUG_BBOX        = True   # YOLO bounding boxes + confidence labels
DEBUG_AUDIO       = False   # lip contour + MAR label + MIC dot
DEBUG_MEDIAPIPE   = True   # iris/eye points, nose dot, H/V lines,
                            # yaw/pitch/gaze values, EAR + blink count

# ── Detection Toggles ────────────────────────────────────────────────────────
# Set False to completely skip a detection — no alert, no processing, no drawing.

# Head / gaze
DETECT_LOOKING_AWAY    = True    # head rotated sideways (yaw)
DETECT_LOOKING_DOWN    = True    # head tilted down (pitch)
DETECT_LOOKING_UP      = True    # head tilted up (pitch)
DETECT_LOOKING_SIDE    = True    # iris gaze left or right
DETECT_FACE_HIDDEN     = True    # person present but face landmarks absent
DETECT_PARTIAL_FACE    = True    # face too small / far from camera
DETECT_FAKE_PRESENCE   = True    # no blink / low head movement (liveness)
DETECT_SPEAKER_AUDIO   = True    # speech active while lips are not moving

# Objects
DETECT_PHONE           = True
DETECT_BOOK            = True
DETECT_HEADPHONE       = True
DETECT_EARBUD          = True
DETECT_MULTIPLE_PEOPLE = True

# ── Drawing (always-on when True, independent of DEBUG) ──────────────────────
DRAW_ALERTS       = True   # warn/alert banners
DRAW_RISK_OVERLAY = True   # score + state panel (top-right)

# ── Alert Cooldowns ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS       = 3    # seconds before the same alert can fire again
RESET_COOLDOWN_SECONDS = 1    # seconds after condition clears before re-arm

# ── Duration Thresholds (seconds a condition must persist before alerting) ───
LOOKING_AWAY_THRESHOLD = 2.0   # head pose conditions (yaw, pitch, face_hidden, etc.)
GAZE_THRESHOLD         = 1.5   # iris gaze conditions (looking_side) — faster trigger

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

# ── Lip Movement / Speaker Audio ─────────────────────────────────────────────
LIP_MAR_SPEAKING    = 0.05
LIP_MAR_YAWN        = 0.22
LIP_YAWN_DURATION_S = 1.5
LIP_DYNAMIC_STD_MIN = 0.010
LIP_HISTORY         = 30

AUDIO_SAMPLE_RATE   = 16_000
AUDIO_CHANNELS      = 1
AUDIO_CHUNK_SAMPLES = 512
AUDIO_SPEECH_THRESH = 0.5
SPEAKER_HOLD_S      = 1.5

# ── Object Detection (YOLO confidence thresholds) ────────────────────────────
YOLO_DEFAULT_CONF = 0.50   # fallback for any class not listed below
YOLO_PERSON_CONF  = 0.30   # lower — person detection should be sensitive
YOLO_PHONE_CONF   = 0.65
YOLO_BOOK_CONF    = 0.70
YOLO_AUDIO_CONF   = 0.41   # shared by headphone + earbud

# ── Object Detection (temporal stability) ────────────────────────────────────
OBJECT_WINDOW    = 15   # rolling frame window
OBJECT_MIN_VOTES = 5    # object must appear in N of last OBJECT_WINDOW frames
PHONE_MIN_VOTES  = 9    # phone  must appear in  9/15 frames (reduces false positives)
BOOK_MIN_VOTES   = 10   # book   must appear in 10/15 frames (high false-positive rate)
EARBUD_MIN_VOTES = 9   # earbud must appear in 9/15 frames (high false-positive rate)

# ── Risk Scoring ─────────────────────────────────────────────────────────────
RISK_SESSION_DURATION_S  = 3600  # assumed exam duration in seconds (used for decay interval)
TIMER_FLICKER_GRACE_S    = 1.5   # seconds condition can be absent before resetting continuous timers

# ── Session Report ────────────────────────────────────────────────────────────
SAVE_REPORT = True        # write a JSON report when the session ends
REPORT_DIR  = "reports"   # folder where report files are saved

# ── Proof capture ─────────────────────────────────────────────────────────────
# Set SAVE_PROOF = False to disable all proof file writing.
# Proof files are saved under REPORT_DIR/proof/.
SAVE_PROOF       = True    # master toggle for proof capture
PROOF_PRE_S      = 2.5    # seconds of video before the alert event
PROOF_POST_S     = 2.5    # seconds of video after the alert event
PROOF_VIDEO_FPS  = 20.0   # output video frame-rate for video/AV clips
