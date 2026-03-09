DEBUG = True

COOLDOWN_SECONDS = 3
RESET_COOLDOWN_SECONDS = 1
LOOKING_AWAY_THRESHOLD = 1.5

SAMPLE_INTERVAL = 0.2
FAKE_WINDOW = 15.0
MIN_VARIANCE = 0.001
NO_BLINK_TIMEOUT = 10

LIVENESS_WEIGHTS = {
    "yaw": 0.45,
    "gaze": 0.45,
    "pitch": 0.10,
}

OBJECT_WINDOW = 15        # frames
OBJECT_MIN_VOTES = 5      # must appear in 5 of last 15 frames

# Audio proctoring
AUDIO_SR = 16000          # sample rate (Hz)
AUDIO_CHUNK = 512         # samples per chunk (~32ms @ 16kHz)
ENROLLMENT_WAV = "enrollment.wav"