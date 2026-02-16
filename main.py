import cv2

from config import *
from utils import AlertManager, draw_alerts, draw_detections
from detectors import ObjectDetector, merge_by_class, HeadPoseDetector
from core import AlertEngine, HeadTracker, LivenessDetector, ObjectTemporalTracker
from collections import Counter

draw_objects = [True,True] #head , objects

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
    "fake_presence": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible fake presence detected (no eye blink / low movement)"}
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
            _
        ) = head_pose_detector.detect(frame, draw=draw_objects[0])

        #Liveness
        liveness.update(yaw, pitch, gaze, blinked)
        fake, _ = liveness.is_fake()

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
            "fake_presence": fake
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
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()