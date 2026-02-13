import cv2
import math

import mediapipe as mp
# Mediapipe : face landmark detection, gives 468 facial points



class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh #This module detects facial landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh( #Creates the actual face detector. This loads the ML model into memory
            static_image_mode = False, #False = video mode. Enables tracking across frames. If True → every frame treated separately (slower)
            max_num_faces=1, #Detect only one face
            refine_landmarks=True, # Enables high-precision landmarks, Enables high-precision landmarks
            min_detection_confidence=0.5, #Minimum confidence to detect face
            min_tracking_confidence=0.5 #Confidence needed to track face between frames : Avoids flickering
        ) 
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

        self.MIN_FACE_WIDTH = 80
        self.MIN_FACE_HEIGHT = 100


        # Eye landmarks (MediaPipe)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Blink config
        self.EAR_THRESHOLD = 0.20 #(Eye Aspect Ratio) : measures how open the eye is
        self.BLINK_FRAMES = 2

        self.blink_counter = 0
        self.total_blinks = 0

    
    def _euclidean(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx*dx + dy*dy) ** 0.5



    def _eye_aspect_ratio(self, eye_points):
        A = self._euclidean(eye_points[1], eye_points[5])
        B = self._euclidean(eye_points[2], eye_points[4])
        C = self._euclidean(eye_points[0], eye_points[3])

        return (A + B) / (2.0 * C)


    def detect(self, frame, draw=True):
        """
        Returns:
            - looking_away (bool)
            - yaw_ratio (float)
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV uses BGR -> MediaPipe needs RGB

        #Run the model
        results = self.face_mesh.process(rgb) #finds face, computes 468 landmarks, stores them in results

        #no face detected
        if not results.multi_face_landmarks:
            self.blink_counter = 0
            return False, False, False, False, False, False,0.0, 0.0, 0.0,0,False,0
        
        """
        multi_face_landmarks → list of faces
        [0] → first face
        .landmark → list of 468 points and Each point has : .x , .y , .z   (normalized 0-1)
        """
        landmarks = results.multi_face_landmarks[0].landmark

        #CONVERT TO PIXELS
        nose_x = int(landmarks[self.NOSE_TIP].x * w) #int() → OpenCV requires integers
        nose_y = int(landmarks[self.NOSE_TIP].y * h)
        left_x = int(landmarks[self.LEFT_CHEEK].x * w)
        right_x = int(landmarks[self.RIGHT_CHEEK].x * w)

        forehead_y = int(landmarks[self.FOREHEAD].y * h)
        chin_y = int(landmarks[self.CHIN].y * h)

        # LEFT EYE
        le_left_x = int(landmarks[self.LEFT_EYE_LEFT].x * w)
        le_right_x = int(landmarks[self.LEFT_EYE_RIGHT].x * w)
        le_iris_x = int(landmarks[self.LEFT_IRIS].x * w)
        le_iris_y = int(landmarks[self.LEFT_IRIS].y * h)

        # RIGHT EYE
        re_left_x = int(landmarks[self.RIGHT_EYE_LEFT].x * w)
        re_right_x = int(landmarks[self.RIGHT_EYE_RIGHT].x * w)
        re_iris_x = int(landmarks[self.RIGHT_IRIS].x * w)
        re_iris_y = int(landmarks[self.RIGHT_IRIS].y * h)


        # Left eye gaze
        left_eye_center = (le_left_x + le_right_x) // 2
        left_eye_width = max(1, le_right_x - le_left_x)
        left_gaze = (le_iris_x - left_eye_center) / left_eye_width

        # Right eye gaze
        right_eye_center = (re_left_x + re_right_x) // 2
        right_eye_width = max(1, re_right_x - re_left_x)
        right_gaze = (re_iris_x - right_eye_center) / right_eye_width

        face_center_y = (forehead_y + chin_y) // 2
        face_height = max(1, chin_y - forehead_y)

        face_center_x = (left_x + right_x) // 2 #Finds midpoint of cheeks. This is horizontal face center
        face_width = max(1,right_x-left_x)


        left_eye = []
        right_eye = []

        for i in self.LEFT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            left_eye.append((x, y))

        for i in self.RIGHT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            right_eye.append((x, y))

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0
        blinked = False

        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_FRAMES:
                self.total_blinks += 1
                blinked = True

            self.blink_counter = 0


        """
        If student moves closer to camera:Face looks bigger & Nose moves more in pixels
        If student moves away: Face looks smaller & Nose moves less

        So raw pixel distance is unreliable. We normalize using face width.
        """
        partial_face = (face_width < self.MIN_FACE_WIDTH or face_height < self.MIN_FACE_HEIGHT)

        yaw_ratio = (nose_x - face_center_x) / face_width #How far nose is from face center / face width
        pitch_ratio = (nose_y - face_center_y) / face_height
        gaze_ratio = (left_gaze + right_gaze) / 2

        """
        How much is the head turned
        normalized horizontal displacement of the nose

        yaw_ratio measures the horizontal displacement of the nose from the face center, 
        normalized by face width, which estimates how much the head is rotated left or right
        
        ≈0  Straight
        +	Looking right
        -	Looking left
        """

        #threshold 
        LOOK_AWAY_YAW = 0.2

        LOOK_DOWN_PITCH = 0.13
        LOOK_UP_PITCH = -0.1

        GAZE_LEFT = -0.15
        GAZE_RIGHT = 0.15


        looking_away = abs(yaw_ratio) > LOOK_AWAY_YAW

        looking_down = pitch_ratio > LOOK_DOWN_PITCH
        looking_up = pitch_ratio < LOOK_UP_PITCH

        looking_left = gaze_ratio < GAZE_LEFT
        looking_right = gaze_ratio > GAZE_RIGHT

        if draw:
            #Nose
            cv2.circle(frame, (nose_x, nose_y), 5, (0,255,255), -1)
            # Left iris
            cv2.circle(frame, (le_iris_x, le_iris_y), 3, (255, 0, 255), -1)
            # Right iris
            cv2.circle(frame, (re_iris_x, re_iris_y), 3, (255, 0, 255), -1)


            #Face Center line
            cv2.line(
                frame,
                (left_x, face_center_y),
                (right_x, face_center_y),
                (0,255,0),
                2 
            )

            #pitch
            cv2.line(
                frame,
                (face_center_x, forehead_y),
                (face_center_x, chin_y),
                (255, 255, 0),
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

            #pitch text
            cv2.putText(
                frame,
                f"Pitch: {pitch_ratio:.2f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0) if not (looking_down or looking_up) else (0, 0, 255),
                2
            )

            #Gaze text
            cv2.putText(
                frame,
                f"Gaze: {gaze_ratio:.2f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255) if not (looking_left or looking_right) else (0, 0, 255),
                2
            )

            # Draw eyes
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (255, 0, 255), -1)

            # EAR + Blink info
            cv2.putText(
                frame,
                f"EAR: {ear:.2f} | Blinks: {self.total_blinks}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )




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
