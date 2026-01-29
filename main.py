import cv2
import time
from utils import AlertManager
from utils import draw_alerts, draw_detections
from detectors import ObjectDetector, merge_person_detections, HeadPoseDetector


COOLDOWN_SECONDS = 3
RESET_COOLDOWN_SECONDS = 1
LOOKING_AWAY_THRESHOLD_SECONDS = 1.5

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")
    
    alert_manager = AlertManager()
    detector = ObjectDetector()
    head_pose_detector = HeadPoseDetector()

    states = {
    "phone" : {"active":False, "last_alert":0},
    "multiple_people" : {"active":False, "last_alert":0},
    "no_person" : {"active":False, "last_alert":0},
    "book" : {"active":False, "last_alert":0},
    "looking_away": {"active": False, "last_alert": 0, "start_time": None}
    }

    def trigger(alert_key, condition, message):
        now = time.time()
        state = states[alert_key]

        if condition:
            if (not state["active"] or (now - state["last_alert"]) > COOLDOWN_SECONDS):
                alert_manager.add_alert(message)
                state["active"] = True
                state["last_alert"] = now
                
        if not condition and state["active"]:
            if (now - state["last_alert"]) > RESET_COOLDOWN_SECONDS:
                state["active"] = False


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        detections = merge_person_detections(detections, iou_threshold=0.5)

        looking_away, yaw_ratio = head_pose_detector.detect(frame, draw=True)
        la_state = states["looking_away"]
        now = time.time()

        if looking_away:
            if la_state["start_time"] is None:
                la_state["start_time"] = now

            duration = now - la_state["start_time"]

            if duration >= LOOKING_AWAY_THRESHOLD_SECONDS:
                trigger("looking_away", True, "ALERT: Looking away from screen")
        else:
            la_state["start_time"] = None
            la_state["active"] = False

        if la_state["start_time"] is not None:
            elapsed = now - la_state["start_time"]
            cv2.putText(
                    frame,
                    f"Away: {elapsed:.1f}s",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
            )        

        draw_detections(frame, detections)

        phone_detected = any(d["class"] == "cell phone" for d in detections)
        people_count = sum(1 for d in detections if d["class"] == "person")
        book_detected = any(d["class"] == "book" for d in detections)

        trigger("phone", phone_detected, "ALERT: Mobile phone detected")
        trigger("book", book_detected, "ALERT: Book detected")
        trigger("multiple_people", people_count > 1, "ALERT: Multiple people detected")
        trigger("no_person", people_count == 0, "ALERT: No person present")

        active_alerts = alert_manager.get_active_alerts()
        draw_alerts(frame, active_alerts)

        cv2.imshow("AI Proctoring Vision POC", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
