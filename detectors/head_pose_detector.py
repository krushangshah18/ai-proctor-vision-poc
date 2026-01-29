import cv2
import mediapipe as mp

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode = False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454

    def detect(self, frame, draw=True):
        """
        Returns:
            - looking_away (bool)
            - yaw_ratio (float)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, 0.0
        
        landmarks = results.multi_face_landmarks[0].landmark

        nose_x = int(landmarks[self.NOSE_TIP].x * w)
        nose_y = int(landmarks[self.NOSE_TIP].y * h)
        left_x = int(landmarks[self.LEFT_CHEEK].x * w)
        right_x = int(landmarks[self.RIGHT_CHEEK].x * w)

        face_center_x = (left_x + right_x) // 2
        face_width = max(1,right_x-left_x)

        yaw_ratio = (nose_x - face_center_x) / (right_x - left_x)
        #threshold 
        looking_away = abs(yaw_ratio) > 0.2

        if draw:
            #Nose
            cv2.circle(frame, (nose_x, nose_y), 5, (0,255,255), -1)

            #Face Center line
            cv2.line(
                frame,
                (face_center_x, nose_y-40),
                (face_center_x, nose_y+40),
                (0,255,0),
                2 
            )

            #Yaw Text
            cv2.putText(
                frame,
                f"Yaw: {yaw_ratio:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if not looking_away else (0,0,255),
                2
            )

        return looking_away, yaw_ratio