import cv2

from config import *
from utils import AlertManager, draw_alerts, draw_detections
from detectors import ObjectDetector, merge_person_detections, HeadPoseDetector
from core import AlertEngine, HeadTracker, LivenessDetector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")

    states = {
    "phone" : {"active":False, "last_alert":0, "message":"ALERT: Mobile phone detected"},
    "multiple_people" : {"active":False, "last_alert":0, "message":"ALERT: Multiple people detected"},
    "no_person" : {"active":False, "last_alert":0, "message":"ALERT: No person present"},
    "book" : {"active":False, "last_alert":0, "message":"ALERT: Book detected"},
    
    "looking_away": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Candidate is not facing the screen"},
    "looking_down": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Candidate is looking down for extended duration"},
    "looking_side": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Candidate is looking away from the screen (eye gaze detected)"},
    "face_hidden": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face not clearly visible (possible obstruction)"},
    "partial_face": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face appears too small (candidate may be too far from camera)"},
    "fake_presence": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Possible fake presence detected (no eye blink / low movement)"}
    }

    alert_manager = AlertManager()
    detector = ObjectDetector()
    head_pose_detector = HeadPoseDetector()

    alerts = AlertEngine(alert_manager, states, COOLDOWN_SECONDS, RESET_COOLDOWN_SECONDS)
    tracker = HeadTracker(states, LOOKING_AWAY_THRESHOLD)
    liveness = LivenessDetector(FAKE_WINDOW, SAMPLE_INTERVAL, MIN_VARIANCE, NO_BLINK_TIMEOUT, LIVENESS_WEIGHTS)


    def track_and_alert(frame, key, condition):
        triggered = tracker.process(frame, key, condition)
        alerts.trigger(key, triggered)


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = merge_person_detections(detector.detect(frame), iou_threshold=0.5)

        (
            looking_away,
            looking_down,
            _,
            looking_left,
            looking_right,
            partial_face,
            yaw,
            pitch,
            gaze,
            _,
            blinked,
            _

        ) = head_pose_detector.detect(frame, draw=True)

        #Liveness
        liveness.update(yaw, pitch, gaze, blinked)
        fake, var_ = liveness.is_fake()
        track_and_alert(frame, "fake_presence", fake) 

        #Head Movement
        track_and_alert(frame, "looking_away", looking_away)
        track_and_alert(frame, "looking_down", looking_down)
        track_and_alert(frame, "looking_side", looking_left or looking_right)
        track_and_alert(frame, "partial_face", partial_face)
        track_and_alert(frame, "face_hidden", (yaw == 0.0 and pitch == 0.0 and gaze == 0.0))

        #Objects
        phone = any(d["class"] == "cell phone" for d in detections)
        book = any(d["class"] == "book" for d in detections)
        people_count = sum(1 for d in detections if d["class"] == "person")

        alerts.trigger("phone", phone)
        alerts.trigger("book", book)
        alerts.trigger("multiple_people", people_count > 1)
        alerts.trigger("no_person", people_count == 0)


        draw_detections(frame, detections)
        draw_alerts(frame, alert_manager.get_active_alerts())

        cv2.imshow("AI Proctor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()