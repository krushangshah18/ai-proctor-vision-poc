import cv2
import math
import mediapipe as mp

class HeadPoseDetector:
    def __init__(self, debug=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh( #Creates the actual face detector.
            static_image_mode = False, #False = video mode. Enables tracking across frames.
            max_num_faces=1, #Detect only one face
            refine_landmarks=True, # Enables high-precision landmarks
            min_detection_confidence=0.5, #Minimum confidence to detect face
            min_tracking_confidence=0.5 #Confidence needed to track face between frames : Avoids flickering
        ) 
        self.DEBUG = debug

        #IDs of specific face points
        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234
        self.RIGHT_CHEEK = 454
        self.FOREHEAD = 10
        self.CHIN = 152

        # Left eye
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.LEFT_IRIS = 468

        # Right eye
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        self.RIGHT_IRIS = 473

        # Face Size Constraints
        self.MIN_FACE_WIDTH = 80
        self.MIN_FACE_HEIGHT = 100


        # Eye landmarks (MediaPipe)
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

        # Blink config
        self.EAR_THRESHOLD = 0.20 #(Eye Aspect Ratio) : measures how open the eye is
        self.BLINK_FRAMES = 2
        self.blink_counter = 0
        self.total_blinks = 0

        # Head  Pose Thresholds
        self.LOOK_AWAY_YAW = 0.2
        self.LOOK_DOWN_PITCH = 0.13
        self.LOOK_UP_PITCH = -0.1
        self.GAZE_LEFT = -0.15
        self.GAZE_RIGHT = 0.15
    

    # Utility Functions
    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _eye_aspect_ratio(self, eye):
        A = self._dist(eye[1], eye[5])
        B = self._dist(eye[2], eye[4])
        C = self._dist(eye[0], eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    def detect(self, frame, draw=True):
        """
        Returns:
            - looking_away (bool)
            - yaw_ratio (float)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV uses BGR -> MediaPipe needs RGB
        
        #Run the model
        results = self.face_mesh.process(rgb) #finds face, computes 468 landmarks, stores them in results

        #no face detected
        if not results.multi_face_landmarks:
            self.blink_counter = 0
            return (False, False, False, False, False, False,0.0, 0.0, 0.0,0,False,0)
        
        """
        multi_face_landmarks → list of faces
        [0] → first face
        .landmark → list of 468 points and Each point has : .x , .y , .z   (normalized 0-1)
        """
        landmarks = results.multi_face_landmarks[0].landmark

        # Convert only required points
        def px(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)
        
        nose = px(self.NOSE_TIP)
        left_cheek = px(self.LEFT_CHEEK)
        right_cheek = px(self.RIGHT_CHEEK)
        forehead = px(self.FOREHEAD)
        chin = px(self.CHIN)

        # Face geometry
        face_width = max(1, right_cheek[0] - left_cheek[0])
        face_height = max(1, chin[1] - forehead[1])
        face_center_x = (left_cheek[0] + right_cheek[0]) // 2
        face_center_y = (forehead[1] + chin[1]) // 2

        partial_face = (
            face_width < self.MIN_FACE_WIDTH or
            face_height < self.MIN_FACE_HEIGHT
        )

        # Head Pose
        """
        raw pixel distance is unreliable. We normalize using face width.
        
        yaw_ratio measures the horizontal displacement of the nose from the face center, 
        normalized by face width, which estimates how much the head is rotated left or right
        
        ≈0  Straight
        +	Looking right
        -	Looking left
        """
        yaw_ratio = (nose[0] - face_center_x) / face_width
        pitch_ratio = (nose[1] - face_center_y) / face_height

        looking_away = abs(yaw_ratio) > self.LOOK_AWAY_YAW
        looking_down = pitch_ratio > self.LOOK_DOWN_PITCH
        looking_up = pitch_ratio < self.LOOK_UP_PITCH

        # Gaze
        le_left = px(self.LEFT_EYE_LEFT)
        le_right = px(self.LEFT_EYE_RIGHT)
        le_iris = px(self.LEFT_IRIS)

        re_left = px(self.RIGHT_EYE_LEFT)
        re_right = px(self.RIGHT_EYE_RIGHT)
        re_iris = px(self.RIGHT_IRIS)

        left_eye_width = max(1, le_right[0] - le_left[0])
        right_eye_width = max(1, re_right[0] - re_left[0])

        left_gaze = (le_iris[0] - (le_left[0] + le_right[0]) // 2) / left_eye_width
        right_gaze = (re_iris[0] - (re_left[0] + re_right[0]) // 2) / right_eye_width
        gaze_ratio = (left_gaze + right_gaze) / 2

        looking_left = gaze_ratio < self.GAZE_LEFT
        looking_right = gaze_ratio > self.GAZE_RIGHT

        # Blink Detection
        left_eye = [px(i) for i in self.LEFT_EYE_POINTS]
        right_eye = [px(i) for i in self.RIGHT_EYE_POINTS]

        ear = (
            self._eye_aspect_ratio(left_eye) +
            self._eye_aspect_ratio(right_eye)
        ) / 2.0

        blinked = False

        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_FRAMES:
                self.total_blinks += 1
                blinked = True

            self.blink_counter = 0

        if draw and self.DEBUG:
            #Nose
            cv2.circle(frame, nose, 4, (0,255,255), -1)

            # Left iris
            cv2.circle(frame, le_iris, 3, (255, 0, 255), -1)
            # Right iris
            cv2.circle(frame, re_iris, 3, (255, 0, 255), -1)

            # Draw eyes
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (255, 0, 255), -1)

            #Face Center line
            cv2.line(
                frame,
                (left_cheek[0], face_center_y),
                (right_cheek[0], face_center_y),
                (0,255,0),
                2 
            )

            #pitch
            cv2.line(
                frame,
                (face_center_x, forehead[1]),
                (face_center_x, chin[1]),
                (255, 255, 0),
                2
            )

            #Yaw Text
            cv2.putText(frame, f"Yaw: {yaw_ratio:.2f}",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0,255,0) if not looking_away else (0,0,255), 2)

            #pitch text
            cv2.putText(frame, f"Pitch: {pitch_ratio:.2f}",
                        (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

            #Gaze text
            cv2.putText(frame, f"Gaze: {gaze_ratio:.2f}",
                        (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

            # EAR + Blink info
            cv2.putText(frame, f"EAR: {ear:.2f} | Blinks: {self.total_blinks}",
                        (20,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,0,255), 2)


        return (
            looking_away,
            looking_down,
            looking_up,
            looking_left,
            looking_right,
            partial_face,
            yaw_ratio,
            pitch_ratio,
            gaze_ratio,
            ear,
            blinked,
            self.total_blinks
        )