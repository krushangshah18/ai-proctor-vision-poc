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
    "phone" : {"active":False, "last_alert":0, "message":"ALERT: Mobile phone detected"},
    "multiple_people" : {"active":False, "last_alert":0, "message":"ALERT: Multiple people detected"},
    "no_person" : {"active":False, "last_alert":0, "message":"ALERT: No person present"},
    "book" : {"active":False, "last_alert":0, "message":"ALERT: Book detected"},
    "looking_away": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Looking away from screen"},
    "looking_down": {"active": False, "last_alert": 0, "start_time": None, "message":"ALERT: Looking down for long time"},
    "looking_side": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Looking away (eye gaze)"},
    "face_hidden": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face not clearly visible / Camera blocked"},
    "partial_face": {"active": False, "last_alert": 0, "start_time": None, "message": "ALERT: Face not fully visible"},
    }

    def trigger(alert_key, condition):
        now = time.time()
        state = states[alert_key]

        if condition:
            if (not state["active"] or (now - state["last_alert"]) > COOLDOWN_SECONDS):
                alert_manager.add_alert(state["message"])
                state["active"] = True
                state["last_alert"] = now
                
        if not condition and state["active"]:
            if (now - state["last_alert"]) > RESET_COOLDOWN_SECONDS:
                state["active"] = False

    def head_movement(frame, state_key, looking_Condition):
        now = time.time()
        this_state = states[state_key]
        label = state_key.replace("_", " ").title()

        if looking_Condition:

            if this_state["start_time"] is None:
                this_state["start_time"] = now

            duration = now - this_state["start_time"]

            if duration >= LOOKING_AWAY_THRESHOLD_SECONDS:
                trigger(state_key, True)
        else:
            this_state["start_time"] = None
            this_state["active"] = False

        if this_state["start_time"] is not None:
            elapsed = now - this_state["start_time"]
            cv2.putText(
                    frame,
                    f"{label}: {elapsed:.1f}s",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
            ) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        detections = merge_person_detections(detections, iou_threshold=0.5)

        (
            looking_away,
            looking_down,
            looking_up,
            looking_left,
            looking_right,
            partial_face,
            yaw_ratio,
            pitch_ratio,
            gaze_ratio
        ) = head_pose_detector.detect(frame, draw=True)

        side_gaze = looking_left or looking_right
        invalid_face = (yaw_ratio == 0.0 and pitch_ratio == 0.0 and gaze_ratio == 0.0)

        head_movement(frame, "looking_away", looking_away)
        head_movement(frame, "looking_down", looking_down)
        head_movement(frame, "looking_side", side_gaze)
        head_movement(frame, "face_hidden", invalid_face)
        head_movement(frame, "partial_face", partial_face)
        
        draw_detections(frame, detections)

        phone_detected = any(d["class"] == "cell phone" for d in detections)
        people_count = sum(1 for d in detections if d["class"] == "person")
        book_detected = any(d["class"] == "book" for d in detections)

        trigger("phone", phone_detected)
        trigger("book", book_detected)
        trigger("multiple_people", people_count > 1)
        trigger("no_person", people_count == 0)
        
        active_alerts = alert_manager.get_active_alerts()
        draw_alerts(frame, active_alerts)

        cv2.imshow("AI Proctoring Vision POC", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
