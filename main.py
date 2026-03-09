import cv2
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import noisereduce as nr

from config import *
from utils import AlertManager, draw_alerts, draw_detections
from detectors import ObjectDetector, merge_by_class, HeadPoseDetector
from core import AlertEngine, HeadTracker, LivenessDetector, ObjectTemporalTracker
from core.audio_proctoring import ProctorSession, CheatType

draw_objects = [True,True] #head , objects

_AUDIO_EVENT_LABELS = {
    CheatType.IMPERSONATION:  "ALERT [AUDIO]: Voice impersonation detected",
    CheatType.GHOST_VOICE:    "ALERT [AUDIO]: Ghost voice / pre-recorded playback",
    CheatType.EXTRA_SPEAKER:  "ALERT [AUDIO]: Extra speaker detected",
    CheatType.VOICE_MISMATCH: "ALERT [AUDIO]: Voice mismatch with enrolled student",
}


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")

    states = {
    "phone" : {"active":False, "last_alert":0, "message":"ALERT: Mobile phone detected"},
    "multiple_people" : {"active":False, "last_alert":0, "message":"ALERT: Multiple people detected"},
    "no_person" : {"active":False, "last_alert":0, "message":"ALERT: No person present"},
    "book" : {"active":False, "last_alert":0, "message":"ALERT: Book detected"},
    "headphone" : {"active":False, "last_alert":0, "message":"ALERT: Headphone detected"},
    "earbud" : {"active":False, "last_alert":0, "message":"ALERT: Earbud detected"},
    
    "looking_away": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Candidate is not facing the screen"},
    
    "looking_down": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Candidate is looking down for extended duration"},

    "looking_up": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Candidate is looking up for extended duration"},

    "looking_side": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is looking away from the screen (eye gaze detected)"},

    "face_hidden": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face not clearly visible (possible obstruction)"},

    "partial_face": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face appears too small (candidate may be too far from camera)"},

    "fake_presence": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible fake presence detected (no eye blink / low movement)"},

    "speaking": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible Speaking Detected"}
    }

    alert_manager = AlertManager()
    detector = ObjectDetector()
    head_pose_detector = HeadPoseDetector(DEBUG)
    object_tracker = ObjectTemporalTracker(
        window=OBJECT_WINDOW,
        min_votes=OBJECT_MIN_VOTES
    )

    alerts = AlertEngine(alert_manager, states, COOLDOWN_SECONDS, RESET_COOLDOWN_SECONDS)
    tracker = HeadTracker(states, LOOKING_AWAY_THRESHOLD, debug=DEBUG)
    liveness = LivenessDetector(FAKE_WINDOW, SAMPLE_INTERVAL, MIN_VARIANCE, NO_BLINK_TIMEOUT, LIVENESS_WEIGHTS)

    # ── Audio proctoring setup ────────────────────────────────────────────────
    session = ProctorSession(sample_rate=AUDIO_SR)
    audio_chunk_q: queue.Queue = queue.Queue(maxsize=100)
    audio_event_q: queue.Queue = queue.Queue()
    stop_audio = threading.Event()
    recorded_chunks: list = []

    # Noise profile: accumulate first 0.5s of mic audio before any speech starts
    _NOISE_PROFILE_CHUNKS = int(0.5 * AUDIO_SR / AUDIO_CHUNK)  # ~16 chunks
    _noise_buf: list = []
    _noise_profile: np.ndarray | None = None

    def _denoise(chunk: np.ndarray) -> np.ndarray:
        nonlocal _noise_profile
        if _noise_profile is None:
            return chunk
        return nr.reduce_noise(y=chunk, sr=AUDIO_SR,
                               y_noise=_noise_profile,
                               stationary=True, prop_decrease=0.85).astype(np.float32)

    def _audio_sd_callback(indata, frames, time_info, status):
        try:
            audio_chunk_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop on overflow rather than block the audio callback

    def _audio_worker():
        nonlocal _noise_profile
        while not stop_audio.is_set():
            try:
                chunk = audio_chunk_q.get(timeout=0.1)

                # build noise profile from the first few silent chunks
                if _noise_profile is None:
                    _noise_buf.append(chunk)
                    if len(_noise_buf) >= _NOISE_PROFILE_CHUNKS:
                        _noise_profile = np.concatenate(_noise_buf)
                        print(f"[Audio] Noise profile captured ({len(_noise_profile)/AUDIO_SR:.2f}s)")
                    recorded_chunks.append(chunk)
                    continue  # don't push to session until profile is ready

                clean = _denoise(chunk)
                recorded_chunks.append(clean)
                events = session.push(clean)
                for ev in events:
                    audio_event_q.put(ev)
            except queue.Empty:
                continue

    # ── Enrollment from wav file ──────────────────────────────────────────────
    try:
        enroll_audio, enroll_sr = sf.read(ENROLLMENT_WAV, dtype="float32")
        if enroll_audio.ndim > 1:
            enroll_audio = enroll_audio[:, 0]
        # Denoise enrollment audio so its embedding lives in the same clean space
        enroll_audio = nr.reduce_noise(y=enroll_audio, sr=enroll_sr,
                                       stationary=True, prop_decrease=0.85).astype(np.float32)
        session.enroll(enroll_audio, sr=enroll_sr)
        print(f"[Audio] Enrolled from {ENROLLMENT_WAV} ({len(enroll_audio) / enroll_sr:.1f}s @ {enroll_sr}Hz)")
    except FileNotFoundError:
        print(f"[Audio] {ENROLLMENT_WAV} not found — audio proctoring disabled")
    except ValueError as e:
        print(f"[Audio] Enrollment failed: {e}")

    # ── Start audio stream and worker thread ──────────────────────────────────
    audio_stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, dtype="float32",
                                  blocksize=AUDIO_CHUNK, callback=_audio_sd_callback)
    audio_stream.start()
    audio_thread = threading.Thread(target=_audio_worker, daemon=True)
    audio_thread.start()

    def track_and_alert(frame, key, condition):
        triggered = tracker.process(frame, key, condition)
        alerts.trigger(key, triggered)


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        raw = detector.detect(frame)

        detections = (merge_by_class(
            raw,
            ["person", "earbud"],
            iou_threshold=0.5
        ) if len(raw) > 1 else raw)


        (
            looking_away,
            looking_down,
            looking_up,
            looking_left,
            looking_right,
            partial_face,
            yaw,
            pitch,
            gaze,
            _,
            blinked,
            _,
            speaking
        ) = head_pose_detector.detect(frame, draw=draw_objects[0])

        #Liveness
        liveness.update(yaw, pitch, gaze, blinked)
        fake, _ = liveness.is_fake()

        # Audio: push current lip state and drain any cheat events
        session.update_lip_activity(speaking)
        while not audio_event_q.empty():
            try:
                ev = audio_event_q.get_nowait()
                label = _AUDIO_EVENT_LABELS.get(ev.event_type, f"ALERT [AUDIO]: {ev.event_type.name}")
                alert_manager.add_alert(f"{label} ({ev.confidence:.0%})")
            except queue.Empty:
                break

        #Object Flags (single pass)
        phone = book = headphone = earbud = False
        people_count = 0

        for d in detections:
            cls = d["class"]
            if cls == "person":
                people_count += 1
            elif cls == "cell_phone":
                phone = True
            elif cls == "book":
                book = True
            elif cls == "headphone":
                headphone = True
            elif cls == "earbud":
                earbud = True

        #Head Movement Conditions
        face_hidden_condition = not (yaw or pitch or gaze) and people_count == 0
        head_conditions = {
            "looking_away": looking_away,
            "looking_down": looking_down,
            "looking_up": looking_up,
            "looking_side": looking_left or looking_right,
            "partial_face": partial_face,
            "face_hidden": face_hidden_condition,
            "fake_presence": fake,
            "speaking": speaking
        }

        for key, cond in head_conditions.items():
            triggered = tracker.process(frame, key, cond)
            alerts.trigger(key, triggered)

        #Object Stability
        object_flags = {
            "phone": phone,
            "book": book,
            "headphone": headphone,
            "earbud": earbud
        }
        for key, present in object_flags.items():
            stable = object_tracker.update(key, present)
            alerts.trigger(key, stable)

        alerts.trigger("multiple_people", people_count > 1)
        # alerts.trigger("no_person", people_count == 0)

        if DEBUG and draw_objects[1]:
            draw_detections(frame, detections)
            draw_alerts(frame, alert_manager.get_active_alerts())
        
        cv2.imshow("AI Proctor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    
    stop_audio.set()
    audio_stream.stop()
    audio_stream.close()

    if recorded_chunks:
        recording = np.concatenate(recorded_chunks)
        sf.write("session_recording.wav", recording, AUDIO_SR)
        print(f"[Audio] Session saved → session_recording.wav ({len(recording) / AUDIO_SR:.1f}s)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()