# Technical Documentation: AI-Based Online Exam Proctoring Detection Pipeline

## 1. System Overview

This repository implements a real-time multimodal exam-proctoring pipeline centered around `main.py`, which acquires webcam frames, runs multiple perception modules, temporally filters their outputs, converts them into scored events, and renders alerts and evidence artifacts for later review. The runtime orchestration is performed in the main loop of [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L175) through [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L473).

At a high level, the system integrates:

- MediaPipe Face Mesh for facial landmark extraction, geometric head-pose heuristics, iris-based gaze estimation, blink detection, lip opening analysis, and liveness cues.
- A YOLO model loaded through Ultralytics for detecting people and prohibited objects such as phones, books, headphones, and earbuds.
- Silero VAD for frame-independent speech activity detection from microphone audio.
- Temporal trackers and a risk engine to convert raw per-frame detections into stable, policy-aware proctoring events.

The complete dataflow is:

1. Webcam frame acquisition.
2. Frame quality check to skip near-black or low-variance frames.
3. YOLO object/person inference.
4. MediaPipe face-landmark inference for pose, gaze, blink, and partial-face analysis.
5. MediaPipe lip-landmark inference for mouth activity.
6. Silero-based audio speech flag readout from an asynchronous audio thread.
7. Temporal stabilization through `HeadTracker`, `ObjectTemporalTracker`, `SpeakerAudioDetector`, and `LivenessDetector`.
8. Event scoring through `RiskEngine`.
9. Warning/alert routing through `AlertEngine`.
10. Optional proof capture through `ProofWriter`.

The most important configuration values are centralized in [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L1) and the risk-policy constants in [settings/scoring.py](/home/krushang/Desktop/ai-proctor-vision-poc/settings/scoring.py#L1).

## 2. Global Inference and Preprocessing Pipeline

### 2.1 Frame acquisition and rejection

The system reads webcam frames using OpenCV through `cv2.VideoCapture(0)` in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L176). Before running the inference stack, each frame is subjected to a simple quality gate:

- Reject the frame if `frame.mean() < 5`
- Reject the frame if `frame.std() < 8`

This logic appears in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L318). It serves as a lightweight safeguard against completely dark frames, camera initialization artifacts, or severe signal loss.

### 2.2 Color preprocessing

Both facial modules convert OpenCV BGR frames to RGB before passing them to MediaPipe:

- Head pose pipeline: [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L86)
- Lip pipeline: [detectors/lip_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/lip_detector.py#L57)

Thus the preprocessing is minimal and online:

\[
I_{RGB} = \text{BGR2RGB}(I_{BGR})
\]

No explicit resizing, histogram normalization, denoising, or cropping is implemented in the current code.

### 2.3 Model concurrency

The video pipeline is synchronous per frame, but audio is processed asynchronously:

- Video inference is done sequentially within the main loop.
- Audio VAD runs in a background thread started by `audio_monitor.start()` in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L271) and implemented in [core/audio_monitor.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/audio_monitor.py#L45).

This design reduces UI blocking from audio capture, but video and MediaPipe inference still share the main thread.

## 3. Module Documentation

## 3.1 Face Detection Using MediaPipe

### 3.1.1 Module Name

`HeadPoseDetector` using `mp.solutions.face_mesh.FaceMesh` in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L14).

### 3.1.2 Purpose of the Module

To detect whether a candidate face is visible and to extract high-density facial landmarks required by downstream pose, gaze, blink, and partial-face logic.

### 3.1.3 Problem It Solves in Exam Proctoring

An online proctoring system must determine whether the candidate is visible, oriented properly, and behaving naturally. Face visibility is the prerequisite for subsequent analyses such as head pose, gaze, speaking verification, and liveness.

### 3.1.4 Algorithmic Approach Used

The implementation does not use a separate face detector API. Instead, it uses MediaPipe Face Mesh with:

- `static_image_mode=False`
- `max_num_faces=1`
- `refine_landmarks=True`
- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`

as defined in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L16).

In practice, this means that face detection and landmark localization are fused into a single Face Mesh pipeline. If `results.multi_face_landmarks` is empty, the system interprets that as face landmarks not being available.

### 3.1.5 Detailed Working Pipeline

1. Capture raw frame from webcam.
2. Convert BGR to RGB.
3. Run `self.face_mesh.process(rgb)`.
4. If no face landmarks are returned, the function outputs all-false states and zero-valued geometric signals.
5. If landmarks are present, selected landmark indices are converted from normalized coordinates to pixel coordinates using frame width and height.

### 3.1.6 Key Mathematical or Logical Formulas

MediaPipe returns normalized landmarks \((x_n, y_n)\in[0,1]\). Pixel conversion is:

\[
x_p = \lfloor x_n \cdot W \rfloor,\quad y_p = \lfloor y_n \cdot H \rfloor
\]

where \(W\) and \(H\) are frame width and height.

### 3.1.7 Model Architecture

The internal MediaPipe architecture is abstracted away by the library. From the repository’s perspective, the module uses a landmark-estimation model that outputs dense face landmarks and iris refinements when `refine_landmarks=True`.

### 3.1.8 Important Parameters and Thresholds Used

- `max_num_faces=1`
- `min_detection_confidence=0.5`
- `min_tracking_confidence=0.5`

from [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L16).

### 3.1.9 Frame Processing Pipeline

The full frame is passed directly to Face Mesh; no face crop is first computed by the application code.

### 3.1.10 Data Flow Between Modules

The face-landmark output is consumed by:

- Head pose estimation.
- Gaze detection.
- Blink detection.
- Partial-face detection.
- Liveness estimation.

Separately, the lip detector runs a second Face Mesh instance for mouth analysis in [detectors/lip_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/lip_detector.py#L43).

### 3.1.11 Edge Cases Handled

- No face landmarks: returns neutral outputs and resets blink counter.
- Landmark tracking mode reduces flicker between consecutive frames.

### 3.1.12 Performance Considerations

The current code runs Face Mesh twice per frame:

- once in `HeadPoseDetector`
- once in `LipDetector`

This is computationally redundant and increases inference latency.

### 3.1.13 Limitations of the Current Implementation

- Face detection is indirect through Face Mesh rather than an explicit face-presence confidence model.
- Only one face is tracked by MediaPipe, although YOLO may detect multiple persons.
- There is no fallback when landmark estimation fails under occlusion, motion blur, or poor illumination.

### 3.1.14 Possible Improvements

- Reuse one shared Face Mesh inference for all face-derived modules.
- Add an explicit face detector for robust face-presence confidence.
- Maintain a face-quality metric to reject low-confidence landmark frames.

## 3.2 Facial Landmark Tracking

### 3.2.1 Module Name

`HeadPoseDetector` and `LipDetector`, both powered by MediaPipe Face Mesh in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L16) and [detectors/lip_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/lip_detector.py#L45).

### 3.2.2 Purpose of the Module

To obtain semantically meaningful landmark coordinates for the nose, cheeks, forehead, chin, eyelids, iris centers, and lips.

### 3.2.3 Problem It Solves in Exam Proctoring

Landmarks enable low-latency geometric analysis without the overhead of training separate models for every suspicious behavior.

### 3.2.4 Algorithmic Approach Used

The implementation accesses specific MediaPipe landmark indices:

- Nose tip: `1`
- Left cheek: `234`
- Right cheek: `454`
- Forehead: `10`
- Chin: `152`
- Eye corners and iris centers: `33,133,468,362,263,473`
- Lip geometry: top `13`, bottom `14`, left `78`, right `308`

These are defined in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L25) and [detectors/lip_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/lip_detector.py#L18).

### 3.2.5 Detailed Working Pipeline

1. Face Mesh returns normalized landmark coordinates.
2. The code converts selected indices to pixel coordinates.
3. Landmark groups are then reused for:
   - face-size estimation
   - pose ratios
   - gaze ratios
   - EAR computation
   - MAR computation

### 3.2.6 Key Mathematical or Logical Formulas

Landmark tracking itself is delegated to MediaPipe. The application performs only deterministic geometric operations on returned points.

### 3.2.7 Model Architecture

Library-provided dense facial landmark model with optional iris refinement.

### 3.2.8 Important Parameters and Thresholds Used

- `refine_landmarks=True` improves precision of eye and iris points.
- `max_num_faces=1` constrains facial landmark tracking to a single candidate face.

### 3.2.9 Frame Processing Pipeline

Tracking occurs frame-by-frame in video mode and relies on MediaPipe’s internal temporal tracking because `static_image_mode=False`.

### 3.2.10 Data Flow Between Modules

Landmark outputs are the shared geometric substrate for head pose, gaze, blink, lip movement, and fake-presence detection.

### 3.2.11 Edge Cases Handled

Missing landmarks produce a neutral tuple in the head detector and `LipState(face_detected=False)` in the lip detector.

### 3.2.12 Performance Considerations

Repeated Face Mesh inference is the primary inefficiency.

### 3.2.13 Limitations of the Current Implementation

- No temporal smoothing is applied directly to landmark coordinates.
- Landmark quality is not explicitly scored.

### 3.2.14 Possible Improvements

- Add landmark smoothing, such as exponential moving averages or Kalman filters.
- Share landmarks between modules.

## 3.3 Head Pose Estimation

### 3.3.1 Module Name

`HeadPoseDetector.detect()` in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L77).

### 3.3.2 Purpose of the Module

To estimate whether the candidate is looking away, looking down, or looking up using simple normalized geometric heuristics.

### 3.3.3 Problem It Solves in Exam Proctoring

Head rotation and tilt are common indicators of off-screen consultation, reading from unauthorized material, or disengagement from the exam interface.

### 3.3.4 Algorithmic Approach Used

The system uses 2D landmark geometry rather than a full Perspective-n-Point head-pose solver. It computes:

- a yaw-like ratio using nose displacement relative to face center and face width
- a pitch-like ratio using nose displacement relative to face center and face height

### 3.3.5 Detailed Working Pipeline

1. Extract pixel coordinates of nose, cheeks, forehead, and chin.
2. Compute face width and height.
3. Estimate face center.
4. Compute normalized pose ratios.
5. Threshold ratios to generate boolean events.
6. Pass those booleans to `HeadTracker` for duration gating.
7. Pass duration-qualified events to `RiskEngine`.

### 3.3.6 Key Mathematical or Logical Formulas

Face geometry:

\[
\text{face\_width} = x_{\text{right cheek}} - x_{\text{left cheek}}
\]

\[
\text{face\_height} = y_{\text{chin}} - y_{\text{forehead}}
\]

Face center:

\[
c_x = \frac{x_{\text{left cheek}} + x_{\text{right cheek}}}{2},\quad
c_y = \frac{y_{\text{forehead}} + y_{\text{chin}}}{2}
\]

Yaw ratio:

\[
\text{yaw\_ratio} = \frac{x_{\text{nose}} - c_x}{\text{face\_width}}
\]

Pitch ratio:

\[
\text{pitch\_ratio} = \frac{y_{\text{nose}} - c_y}{\text{face\_height}}
\]

Decision rules from [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L139):

\[
\text{looking\_away} = |\text{yaw\_ratio}| > 0.20
\]

\[
\text{looking\_down} = \text{pitch\_ratio} > 0.13
\]

\[
\text{looking\_up} = \text{pitch\_ratio} < -0.10
\]

### 3.3.7 Model Architecture

No separate pose model is used. This is a rule-based geometric estimator on top of Face Mesh landmarks.

### 3.3.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L50):

- `LOOK_AWAY_YAW = 0.20`
- `LOOK_DOWN_PITCH = 0.13`
- `LOOK_UP_PITCH = -0.10`
- Duration gate for head-pose events: `LOOKING_AWAY_THRESHOLD = 2.0 s`

### 3.3.9 Frame Processing Pipeline

Per-frame pose booleans are unstable by nature, so they are not used directly. They are fed into `HeadTracker.process()` in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L407), which requires a condition to remain true for a configurable duration before the event becomes active.

### 3.3.10 Data Flow Between Modules

`HeadPoseDetector` outputs:

- pose booleans for `HeadTracker`
- `yaw`, `pitch`, and `gaze` signals for `LivenessDetector`

### 3.3.11 Edge Cases Handled

- `max(1, ...)` is used for width and height denominators to avoid division by zero.
- Missing face landmarks return a neutral output tuple.

### 3.3.12 Performance Considerations

This method is efficient because it avoids full 3D reconstruction or camera calibration.

### 3.3.13 Limitations of the Current Implementation

- The estimator is not true 3D head pose.
- It depends strongly on frontal visibility and stable cheek landmarks.
- It is sensitive to facial asymmetry, camera angle, and partial occlusion.

### 3.3.14 Possible Improvements

- Replace ratios with PnP-based 3D pose estimation.
- Calibrate pose thresholds per camera and user distance.
- Smooth pose signals temporally before thresholding.

## 3.4 Eye Gaze Detection

### 3.4.1 Module Name

Gaze logic inside `HeadPoseDetector.detect()` in [detectors/head_pose_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/head_pose_detector.py#L143).

### 3.4.2 Purpose of the Module

To detect lateral gaze shifts that may indicate the candidate is reading from a secondary screen, notes, or another person.

### 3.4.3 Problem It Solves in Exam Proctoring

Candidates may keep their head relatively stable while shifting only the eyes. A head-pose-only system would miss such behavior.

### 3.4.4 Algorithmic Approach Used

The code estimates normalized iris displacement within each eye by comparing the iris center to the midpoint of the eye corners.

### 3.4.5 Detailed Working Pipeline

1. Extract left and right eye corner coordinates.
2. Extract left and right iris center landmarks.
3. Compute horizontal eye width for each eye.
4. Compute normalized iris displacement from the eye midpoint.
5. Average both eyes into one `gaze_ratio`.
6. Threshold the averaged ratio into `looking_left` or `looking_right`.
7. Merge both into a single `looking_side` event in `main.py`.
8. Apply a shorter duration gate of `1.5 s`.

### 3.4.6 Key Mathematical or Logical Formulas

For each eye:

\[
\text{eye\_mid}_x = \frac{x_{\text{left corner}} + x_{\text{right corner}}}{2}
\]

\[
\text{left\_gaze} = \frac{x_{\text{left iris}} - \text{left\_eye\_mid}_x}{\text{left\_eye\_width}}
\]

\[
\text{right\_gaze} = \frac{x_{\text{right iris}} - \text{right\_eye\_mid}_x}{\text{right\_eye\_width}}
\]

Combined gaze ratio:

\[
\text{gaze\_ratio} = \frac{\text{left\_gaze} + \text{right\_gaze}}{2}
\]

Decision rules:

\[
\text{looking\_left} = \text{gaze\_ratio} < -0.13
\]

\[
\text{looking\_right} = \text{gaze\_ratio} > 0.13
\]

from [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L54).

### 3.4.7 Model Architecture

No dedicated gaze model is used; gaze is derived directly from iris landmarks.

### 3.4.8 Important Parameters and Thresholds Used

- `GAZE_LEFT = -0.13`
- `GAZE_RIGHT = 0.13`
- `GAZE_THRESHOLD = 1.5 s` for event activation

### 3.4.9 Frame Processing Pipeline

The instantaneous gaze ratio is computed per frame and then passed through duration gating in `HeadTracker`.

### 3.4.10 Data Flow Between Modules

`looking_left or looking_right` is mapped into the higher-level event `looking_side` in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L395).

### 3.4.11 Edge Cases Handled

- Eye widths are clamped with `max(1, ...)` to avoid zero division.

### 3.4.12 Performance Considerations

The method is lightweight because it uses already available landmarks.

### 3.4.13 Limitations of the Current Implementation

- Only horizontal gaze is considered.
- There is no compensation for eyelid closure, squinting, or oblique camera position.
- The system does not explicitly reject frames with poor iris visibility.

### 3.4.14 Possible Improvements

- Add vertical gaze estimation.
- Use head-pose compensated gaze.
- Apply a confidence filter based on eye openness or iris visibility.

## 3.5 Speaking Detection Using Mouth Landmarks

### 3.5.1 Module Name

`LipDetector` in [detectors/lip_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/lip_detector.py#L43).

### 3.5.2 Purpose of the Module

To determine whether visible lip dynamics are consistent with speaking.

### 3.5.3 Problem It Solves in Exam Proctoring

The system must distinguish legitimate speech by the candidate from suspicious external audio, such as a speaker or another person helping the candidate.

### 3.5.4 Algorithmic Approach Used

The lip module computes a mouth aspect ratio (MAR) and combines:

- mouth openness
- short-term MAR variability
- yawn suppression

to infer `is_speaking`.

### 3.5.5 Detailed Working Pipeline

1. Run Face Mesh.
2. Extract upper lip, lower lip, left mouth corner, and right mouth corner.
3. Compute vertical mouth opening and horizontal mouth width.
4. Compute MAR.
5. Append MAR to a history buffer of length `LIP_HISTORY=30`.
6. Mark `is_open` when MAR exceeds speaking threshold.
7. Mark `is_yawning` if MAR stays above a higher threshold for a sustained duration.
8. Mark `is_speaking` only if mouth is open, dynamic, and not yawning.

### 3.5.6 Key Mathematical or Logical Formulas

Vertical mouth opening:

\[
v = \|p_{\text{top lip}} - p_{\text{bottom lip}}\|_2
\]

Horizontal mouth width:

\[
h = \|p_{\text{left mouth}} - p_{\text{right mouth}}\|_2
\]

Mouth aspect ratio:

\[
\text{MAR} = \frac{v}{h + \epsilon}
\]

where \(\epsilon = 10^{-6}\) in the code.

Speaking logic:

\[
\text{is\_open} = \text{MAR} > 0.05
\]

\[
\text{dynamic} = \operatorname{std}(\text{last 12 MAR values}) \ge 0.010
\]

\[
\text{is\_speaking} = \text{is\_open} \land \text{dynamic} \land \neg \text{is\_yawning}
\]

Yawn logic:

\[
\text{is\_yawning} = \text{MAR} > 0.22 \text{ for at least } 1.5\text{ s}
\]

### 3.5.7 Model Architecture

MediaPipe Face Mesh followed by rule-based mouth-state classification.

### 3.5.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L76):

- `LIP_MAR_SPEAKING = 0.05`
- `LIP_MAR_YAWN = 0.22`
- `LIP_YAWN_DURATION_S = 1.5`
- `LIP_DYNAMIC_STD_MIN = 0.010`
- `LIP_HISTORY = 30`

### 3.5.9 Frame Processing Pipeline

The mouth signal is computed framewise, but the speaking decision depends on a short sliding MAR history through a `deque`.

### 3.5.10 Data Flow Between Modules

`LipDetector.process()` returns a `LipState` object whose `is_speaking` flag is consumed by `SpeakerAudioDetector` in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L364).

### 3.5.11 Edge Cases Handled

- No face landmarks produce `face_detected=False`.
- Early frames with insufficient MAR history are classified as not dynamic, which avoids premature speaking detection.

### 3.5.12 Performance Considerations

The method is inexpensive, but it duplicates Face Mesh inference already performed by the head detector.

### 3.5.13 Limitations of the Current Implementation

- It uses only mouth opening, not phoneme-level articulation features.
- Quiet speech with low lip displacement may be missed.
- Non-speech mouth motion can still raise MAR variability.

### 3.5.14 Possible Improvements

- Reuse shared landmarks from the head detector.
- Incorporate optical flow around the lips.
- Add a learned audiovisual synchronization model.

## 3.6 Audio Voice Activity Detection Using Silero VAD

### 3.6.1 Module Name

`AudioMonitor` in [core/audio_monitor.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/audio_monitor.py#L12).

### 3.6.2 Purpose of the Module

To detect whether speech-like audio is present in the microphone stream.

### 3.6.3 Problem It Solves in Exam Proctoring

Audio activity may reveal collusion, spoken assistance, or playback from external devices.

### 3.6.4 Algorithmic Approach Used

The code loads Silero VAD via `load_silero_vad()` and runs it continuously on short microphone chunks in a background thread.

### 3.6.5 Detailed Working Pipeline

1. Open microphone stream via PyAudio.
2. Read fixed-size chunks of `512` samples.
3. Convert raw 16-bit PCM to float32 in `[-1,1]`.
4. Convert the NumPy array to a Torch tensor.
5. Run Silero VAD to get speech probability.
6. Threshold the probability at `0.5`.
7. Store:
   - current boolean speech state
   - timestamped raw audio bytes in a ring buffer for proof generation

### 3.6.6 Key Mathematical or Logical Formulas

PCM normalization:

\[
x_{\text{float}} = \frac{x_{\text{int16}}}{32768.0}
\]

Speech decision:

\[
\text{speech\_active} = \mathbb{1}\{p_{\text{speech}} \ge 0.5\}
\]

### 3.6.7 Model Architecture

Silero VAD, a pretrained speech activity detector, is used as a black-box probability estimator.

### 3.6.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L83):

- `AUDIO_SAMPLE_RATE = 16000`
- `AUDIO_CHANNELS = 1`
- `AUDIO_CHUNK_SAMPLES = 512`
- `AUDIO_SPEECH_THRESH = 0.5`
- Audio ring duration default in code: `30 s`

### 3.6.9 Frame Processing Pipeline

This module is not frame-based; it is chunk-based in continuous time. The main video loop reads the latest speech flag through `speech_active()`.

### 3.6.10 Data Flow Between Modules

- The speech flag is combined with lip activity in `SpeakerAudioDetector`.
- The raw audio ring buffer is later queried by `ProofWriter` to generate evidence clips.

### 3.6.11 Edge Cases Handled

- Exceptions inside the audio thread are captured in `self._error`.
- `exception_on_overflow=False` reduces stream failure on timing jitter.

### 3.6.12 Performance Considerations

The threaded design prevents audio capture from blocking the main frame loop.

### 3.6.13 Limitations of the Current Implementation

- It performs only speech activity detection, not speaker identification.
- Environmental speech and candidate speech are not separated.
- The main pipeline currently uses only a boolean flag, discarding the raw probability.

### 3.6.14 Possible Improvements

- Use speech probability smoothing instead of a hard threshold.
- Add speaker diarization or speaker verification.
- Add noise suppression and echo cancellation.

## 3.7 Object Detection Using YOLOv8

### 3.7.1 Module Name

`ObjectDetector` in [detectors/object_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/object_detector.py#L88).

### 3.7.2 Purpose of the Module

To detect people and prohibited objects including mobile phones, books, headphones, and earbuds.

### 3.7.3 Problem It Solves in Exam Proctoring

Visual object evidence is essential for catching unauthorized devices and materials.

### 3.7.4 Algorithmic Approach Used

The system loads a YOLO model from `finalBestV5.pt` via Ultralytics:

\[
\text{detections} = \text{YOLO}(I)
\]

Then, it filters raw predictions by:

- allowed class name
- class-specific confidence threshold

### 3.7.5 Detailed Working Pipeline

1. Initialize YOLO model with the custom weight file.
2. Run inference on the entire frame.
3. Iterate through bounding boxes.
4. Map class IDs to names through `model.names`.
5. Keep only classes in `{"person", "cell_phone", "book", "headphone", "earbud"}`.
6. Apply class-specific confidence gates.
7. Convert boxes to integer `(x1, y1, x2, y2)`.
8. Return detection dictionaries.
9. In `main.py`, optionally merge overlapping `person` and `earbud` detections using IoU-based clustering.

### 3.7.6 Key Mathematical or Logical Formulas

Intersection over Union used in `merge_by_class()`:

\[
\operatorname{IoU}(A,B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
\]

implemented in [detectors/object_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/object_detector.py#L3).

### 3.7.7 Model Architecture

The code uses the Ultralytics `YOLO` interface, but the exact backbone/head details depend on `finalBestV5.pt`, which is a pretrained weight file not programmatically described in the repository. Therefore, the implementation-level documentation can only state that a YOLO-family detector is used through Ultralytics.

### 3.7.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L90):

- `YOLO_DEFAULT_CONF = 0.50`
- `YOLO_PERSON_CONF = 0.30`
- `YOLO_PHONE_CONF = 0.65`
- `YOLO_BOOK_CONF = 0.70`
- `YOLO_AUDIO_CONF = 0.41`

However, the implementation detail matters: `person_conf` is stored but never added to `class_thresholds` in [detectors/object_detector.py](/home/krushang/Desktop/ai-proctor-vision-poc/detectors/object_detector.py#L100), so `person` detections actually use the default threshold of `0.50`, not `0.30`. This is an important code-level inconsistency.

### 3.7.9 Frame Processing Pipeline

Object detection is performed on each accepted frame before higher-level rule evaluation.

### 3.7.10 Data Flow Between Modules

The raw detection list is consumed by:

- object-presence flags for phone, book, headphone, earbud
- person counting for multiple-person detection
- heuristics for `face_hidden` and `no_person`
- `ObjectTemporalTracker` for stability voting

### 3.7.11 Edge Cases Handled

- IoU-based merging reduces duplicate `person` and `earbud` boxes.
- Disallowed classes are filtered out entirely.

### 3.7.12 Performance Considerations

YOLO inference is one of the most computationally expensive parts of the pipeline. Since the code runs inference on every frame with no frame skipping or input resizing strategy in application code, throughput depends heavily on the chosen model size and hardware.

### 3.7.13 Limitations of the Current Implementation

- No explicit NMS customization beyond the model default.
- `person_conf` is not actually applied.
- No object tracking IDs are maintained across frames.

### 3.7.14 Possible Improvements

- Fix the missing `person` threshold mapping.
- Add tracker-assisted temporal association.
- Tune model input size and hardware backend explicitly.

## 3.8 Multiple Person Detection

### 3.8.1 Module Name

The logic in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L430) with policy handling in [core/risk_engine.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/risk_engine.py#L402).

### 3.8.2 Purpose of the Module

To detect when more than one person is visible in the exam frame.

### 3.8.3 Problem It Solves in Exam Proctoring

A second visible person suggests assistance, impersonation, or other policy violations.

### 3.8.4 Algorithmic Approach Used

This module uses YOLO person detections and simply counts the number of remaining `person` boxes after optional box merging.

### 3.8.5 Detailed Working Pipeline

1. YOLO produces `person` detections.
2. Overlapping `person` boxes may be merged with IoU threshold `0.5`.
3. `people_count` is incremented for each `person`.
4. Event condition becomes `people_count > 1`.
5. `RiskEngine.process_event("multiple_people", ...)` handles:
   - occurrence counting
   - warning-only grace on first occurrence
   - non-decaying score escalation on later occurrences
   - automatic exam termination after continuous presence for `20 s`

### 3.8.6 Key Mathematical or Logical Formulas

Decision rule:

\[
\text{multiple\_people} = \mathbb{1}\{N_{\text{person}} > 1\}
\]

Termination rule:

\[
\text{terminate if continuous multiple\_people duration} \ge 20\text{ s}
\]

### 3.8.7 Model Architecture

Depends entirely on the YOLO detector used upstream.

### 3.8.8 Important Parameters and Thresholds Used

From [settings/scoring.py](/home/krushang/Desktop/ai-proctor-vision-poc/settings/scoring.py#L84):

- Grace occurrences: `1`
- Second occurrence score: `20`
- Third and later occurrence score: `50`
- Termination threshold: `20 s`
- Score cooldown: `10 s`

### 3.8.9 Frame Processing Pipeline

Unlike phone/book/earbud, this module does not use the `ObjectTemporalTracker`; it is evaluated directly each frame and handled as a continuous event in the risk engine.

### 3.8.10 Data Flow Between Modules

YOLO person count feeds directly into `RiskEngine`.

### 3.8.11 Edge Cases Handled

Flicker grace of `1.5 s` in `RiskEngine` prevents brief detector dropouts from resetting the continuous timer immediately.

### 3.8.12 Performance Considerations

Person count quality depends on YOLO recall and duplicate suppression.

### 3.8.13 Limitations of the Current Implementation

- No identity tracking; two fragmented boxes can still distort the count.
- Since the practical person threshold is `0.50`, smaller or distant people may be missed more often than intended.

### 3.8.14 Possible Improvements

- Add multi-object tracking and track-based count smoothing.
- Lower or correctly apply the person detection threshold.

## 3.9 Fake Presence Detection

### 3.9.1 Module Name

`LivenessDetector` in [core/liveness.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/liveness.py#L3) plus `fake_presence` integration in [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L398).

### 3.9.2 Purpose of the Module

To detect whether the observed face behaves like a static image or spoof rather than a live person.

### 3.9.3 Problem It Solves in Exam Proctoring

A candidate may attempt to deceive the system using a static photograph or fixed display.

### 3.9.4 Algorithmic Approach Used

The liveness heuristic combines:

- temporal variance of yaw
- temporal variance of pitch
- temporal variance of gaze
- blink absence over time

The system declares fake presence only when the face is both geometrically static and blink-free for a sufficient duration.

### 3.9.5 Detailed Working Pipeline

1. `HeadPoseDetector` outputs `yaw`, `pitch`, `gaze`, and `blinked`.
2. Every `0.2 s`, `LivenessDetector.update()` samples these values.
3. Samples older than `15 s` are dropped.
4. Variance is computed for each signal stream.
5. A weighted variance score is formed.
6. If the weighted score is below `0.001`, the face is considered static.
7. If no blink has been seen for more than `10 s`, `no_blink=True`.
8. The module reports fake presence only if both conditions hold.
9. The `HeadTracker` then requires sustained activation before the event becomes active.
10. The `RiskEngine` uses tiered scoring:
   - warning before `10 s`
   - `30` points after `10 s`
   - `60` points after `25 s`

### 3.9.6 Key Mathematical or Logical Formulas

Variance:

\[
\operatorname{Var}(x) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
\]

Weighted liveness score:

\[
S = 0.45\operatorname{Var}(\text{yaw}) + 0.45\operatorname{Var}(\text{gaze}) + 0.10\operatorname{Var}(\text{pitch})
\]

Static rule:

\[
\text{static} = \mathbb{1}\{S < 0.001\}
\]

Blink timeout rule:

\[
\text{no\_blink} = \mathbb{1}\{t - t_{\text{last blink}} > 10\text{ s}\}
\]

Final decision:

\[
\text{fake} = \text{static} \land \text{no\_blink}
\]

### 3.9.7 Model Architecture

No learned liveness model is used; this is a handcrafted temporal heuristic.

### 3.9.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L69):

- `SAMPLE_INTERVAL = 0.2 s`
- `FAKE_WINDOW = 15.0 s`
- `MIN_VARIANCE = 0.001`
- `NO_BLINK_TIMEOUT = 10 s`
- `LIVENESS_WEIGHTS = {"yaw": 0.45, "gaze": 0.45, "pitch": 0.10}`

Scoring thresholds from [settings/scoring.py](/home/krushang/Desktop/ai-proctor-vision-poc/settings/scoring.py#L134):

- duration tier 1: `10 s`
- duration tier 2: `25 s`

### 3.9.9 Frame Processing Pipeline

This module performs periodic subsampling rather than processing every frame equally. The signal history forms a sliding temporal window.

### 3.9.10 Data Flow Between Modules

Head pose and blink features feed `LivenessDetector`; its boolean output feeds `HeadTracker` and then `RiskEngine`.

### 3.9.11 Edge Cases Handled

If fewer than 10 samples are available, `_variance()` returns `1.0`, intentionally biasing the system toward assuming a real person during startup.

### 3.9.12 Performance Considerations

The module is lightweight because it stores only scalar time series.

### 3.9.13 Limitations of the Current Implementation

- A very still candidate could be misclassified if they do not blink.
- A replayed video with natural motion might evade detection.
- There is no texture-based anti-spoofing.

### 3.9.14 Possible Improvements

- Add texture, depth, or challenge-response liveness checks.
- Model natural micro-motion more robustly.

## 3.10 Proctoring Scoring System

### 3.10.1 Module Name

`RiskEngine` in [core/risk_engine.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/risk_engine.py#L55).

### 3.10.2 Purpose of the Module

To convert raw and temporally stabilized suspicious behaviors into a cumulative risk score and exam state.

### 3.10.3 Problem It Solves in Exam Proctoring

Real-world monitoring requires graded response, not binary decisions. The scoring engine distinguishes between low-severity transient events and repeated or prolonged misconduct.

### 3.10.4 Algorithmic Approach Used

The engine combines:

- rising-edge occurrence counting
- per-event cooldowns
- fixed and decaying score buckets
- duration-tiered scoring
- combo bonuses
- gaze aggregation bonuses
- continuous-duration termination rules

### 3.10.5 Detailed Working Pipeline

1. Each module emits a boolean `active` state.
2. `RiskEngine.process_event()` compares current and previous active states.
3. A rising edge increments the occurrence count.
4. The event is scored only if it passes confidence and cooldown checks.
5. Depending on the event key, the score may be:
   - fixed non-decaying
   - decaying
   - occurrence-based
   - duration-tiered
   - special-episode based in the case of `speaker_audio`
6. The total score updates exam state:
   - `NORMAL`
   - `WARNING`
   - `HIGH_RISK`
   - `ADMIN_REVIEW`
   - `TERMINATED`
7. The decaying bucket is periodically reduced.

### 3.10.6 Key Mathematical or Logical Formulas

Total score:

\[
\text{score} = \text{fixed\_score} + \text{decaying\_score}
\]

Decay interval:

\[
\Delta t_{\text{decay}} = \max(60,\; \frac{\text{session duration}}{20})
\]

With `RISK_SESSION_DURATION_S = 3600`, the current implementation uses:

\[
\Delta t_{\text{decay}} = \max(60,180)=180\text{ s}
\]

Decay update:

\[
\text{decaying\_score} \leftarrow \max(0,\text{decaying\_score}-5)
\]

State thresholds from [settings/scoring.py](/home/krushang/Desktop/ai-proctor-vision-poc/settings/scoring.py#L49):

- `WARNING` at score `>= 30`
- `HIGH_RISK` at score `>= 60`
- `ADMIN_REVIEW` at score `>= 100`

### 3.10.7 Model Architecture

Rule-based decision engine, not a learned model.

### 3.10.8 Important Parameters and Thresholds Used

Representative values from [settings/scoring.py](/home/krushang/Desktop/ai-proctor-vision-poc/settings/scoring.py#L27):

- Gaze events: `5` points each
- Book: `20` decaying
- Headphone/Earbud: `20` decaying after first grace occurrence
- Phone: `25` on second occurrence, `50` on third+
- Multiple people: `20` on second occurrence, `50` on third+
- No person: `25` at 5 s, `50` at 10 s, terminate at 20 s
- Partial face: `2` points after 5 s gate
- Face hidden: `10` at 5 s, `20` at 10 s
- Fake presence: `30` at 10 s, `60` at 25 s

### 3.10.9 Frame Processing Pipeline

Every detection key is processed independently once per loop iteration.

### 3.10.10 Data Flow Between Modules

`RiskEngine` is the fusion hub that consumes all stabilized module outputs and emits `RiskEvent`, which is then consumed by `AlertEngine`.

### 3.10.11 Edge Cases Handled

- Cooldowns prevent score explosion.
- Flicker grace avoids resetting long-duration timers because of brief detector failures.
- Once terminated, further events immediately return a terminated state.

### 3.10.12 Performance Considerations

The scoring engine is computationally negligible relative to vision inference.

### 3.10.13 Limitations of the Current Implementation

- The bucket cap is shared at `150` for both fixed and decaying scores, which may or may not align with intended policy semantics.
- Scoring rules are handcrafted and may need calibration for real deployments.

### 3.10.14 Possible Improvements

- Learn event priors from validation sessions.
- Add per-student or per-camera calibration.
- Externalize risk policy to a report-friendly configuration file or database.

## 3.11 Temporal Event Detection Logic

### 3.11.1 Module Name

Temporal logic is distributed across:

- `HeadTracker` for duration gating in [core/head_tracker.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/head_tracker.py#L18)
- `ObjectTemporalTracker` for sliding-window voting in [core/object_tracker.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/object_tracker.py#L3)
- `SpeakerAudioDetector` for audiovisual desynchronization holding in [core/audio_monitor.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/audio_monitor.py#L96)
- `RiskEngine` for cooldowns, termination timers, and flicker grace in [core/risk_engine.py](/home/krushang/Desktop/ai-proctor-vision-poc/core/risk_engine.py#L140)

### 3.11.2 Purpose of the Module

To reduce frame-level noise and convert transient events into policy-relevant sustained violations.

### 3.11.3 Problem It Solves in Exam Proctoring

Raw detections fluctuate because of motion blur, illumination changes, and model uncertainty. Temporal logic stabilizes the system.

### 3.11.4 Algorithmic Approach Used

Four distinct temporal strategies are used:

1. Duration thresholding for head/gaze/liveness events.
2. Sliding-window voting for object persistence.
3. Hold-time filtering for speaker-audio desynchronization.
4. Cooldown and flicker-grace logic in the risk engine.

### 3.11.5 Detailed Working Pipeline

#### A. Duration gating with `HeadTracker`

For keys such as `looking_away`, `looking_down`, `face_hidden`, `partial_face`, and `fake_presence`, the event becomes active only after continuous truth for a threshold duration.

- Default threshold: `2.0 s`
- Gaze threshold: `1.5 s`

#### B. Sliding-window voting with `ObjectTemporalTracker`

Object flags are stored as `1/0` in a deque of length `15`. An object becomes stable only if votes exceed a configured threshold:

- default: `5/15`
- phone: `9/15`
- book: `10/15`
- earbud: `9/15`

Thus, this module implements a finite-length majority-style persistence test rather than simple consecutive-frame counting.

#### C. Speaker-audio hold logic

In `SpeakerAudioDetector`, audio-lip desynchronization must persist for `0.3 s` before `speaker_flagged=True`.

#### D. Flicker grace in `RiskEngine`

For `multiple_people`, `no_person`, and `speaker_audio`, timers do not reset immediately when the condition becomes false. Instead, the system waits for:

\[
1.5\text{ s}
\]

of continuous inactivity before clearing the episode timer.

### 3.11.6 Key Mathematical or Logical Formulas

Object stability:

\[
\text{stable}(k) = \mathbb{1}\left\{\sum_{i=1}^{15} x_i \ge \tau_k\right\}
\]

where \(x_i \in \{0,1\}\) indicates presence in the \(i\)-th recent frame.

Speaker desynchronization:

\[
\text{desync} = \text{speech\_active} \land (\neg\text{face\_detected} \lor \neg\text{lip\_speaking})
\]

Flag after:

\[
\text{duration(desync)} \ge 0.3\text{ s}
\]

### 3.11.7 Model Architecture

Not a model; this is temporal decision logic.

### 3.11.8 Important Parameters and Thresholds Used

From [config.py](/home/krushang/Desktop/ai-proctor-vision-poc/config.py#L46):

- `LOOKING_AWAY_THRESHOLD = 2.0 s`
- `GAZE_THRESHOLD = 1.5 s`
- `OBJECT_WINDOW = 15`
- `OBJECT_MIN_VOTES = 5`
- `PHONE_MIN_VOTES = 9`
- `BOOK_MIN_VOTES = 10`
- `EARBUD_MIN_VOTES = 9`
- `SPEAKER_HOLD_S = 0.3 s`
- `TIMER_FLICKER_GRACE_S = 1.5 s`

### 3.11.9 Frame Processing Pipeline

Temporal logic sits between raw perception and risk scoring.

### 3.11.10 Data Flow Between Modules

Raw booleans or ratios are transformed into stable booleans before the risk engine sees them.

### 3.11.11 Edge Cases Handled

- `HeadTracker` resets timers immediately when the condition disappears.
- `RiskEngine` retains long-duration state for selected keys despite short dropouts.

### 3.11.12 Performance Considerations

All temporal logic structures are lightweight:

- `deque` for object votes
- scalar timestamps for hold logic
- short lists for liveness histories

### 3.11.13 Limitations of the Current Implementation

- Temporal logic is heterogeneous across modules rather than centrally unified.
- Some keys use `HeadTracker`, others use `RiskEngine`, and objects use their own vote filter, which can complicate calibration.

### 3.11.14 Possible Improvements

- Introduce a common temporal abstraction for all event types.
- Use probabilistic temporal filtering instead of binary thresholds.

## 4. Data Flow Between Modules

The inter-module data flow can be summarized as follows:

1. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L336) runs `ObjectDetector.detect(frame)` to obtain `person`, `cell_phone`, `book`, `headphone`, and `earbud`.
2. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L348) runs `HeadPoseDetector.detect(frame)` to obtain head/gaze booleans and continuous pose features.
3. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L357) forwards `yaw`, `pitch`, `gaze`, and `blinked` to `LivenessDetector`.
4. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L362) runs `LipDetector.process(frame, ts)`.
5. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L363) reads current `speech_active` from `AudioMonitor`.
6. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L365) combines speech and lip state through `SpeakerAudioDetector`.
7. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L407) applies `HeadTracker` to head/gaze/liveness conditions.
8. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L426) applies `ObjectTemporalTracker` to object presence.
9. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L413) and [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L427) send stabilized events to `RiskEngine`.
10. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L277) routes `RiskEvent` through `AlertEngine`.
11. [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L290) optionally stores proof via `ProofWriter`.

## 5. Confidence Thresholds Used in the Implementation

### 5.1 MediaPipe thresholds

- Face Mesh detection confidence: `0.5`
- Face Mesh tracking confidence: `0.5`

### 5.2 YOLO thresholds

- Default class threshold: `0.50`
- Phone: `0.65`
- Book: `0.70`
- Headphone: `0.41`
- Earbud: `0.41`
- Intended person threshold: `0.30`, but the current code effectively uses `0.50`

### 5.3 Scoring confidence thresholds

From `settings/scoring.py`, minimum confidence for score addition:

- global default: `0.5`
- phone: `0.60`
- book: `0.65`
- headphone: `0.40`
- earbud: `0.40`

## 6. Sliding Window, Buffering, and Smoothing Techniques

The implementation uses several buffering mechanisms:

- `ObjectTemporalTracker`: deque-based `15`-frame voting.
- `LipDetector`: `30`-value MAR history.
- `LivenessDetector`: `15 s` sliding window of sampled yaw/pitch/gaze.
- `AudioMonitor`: `30 s` ring buffer of timestamped raw PCM bytes.
- `ProofWriter`: `150`-frame rolling buffer for pre-event video proof.

Notably, the system does not use numerical smoothing such as moving averages for pose or gaze values; it uses event-duration logic instead of signal smoothing.

## 7. Event Detection Rules

The core rule set implemented by the repository is:

- `looking_away`: `|yaw_ratio| > 0.20` for at least `2.0 s`
- `looking_down`: `pitch_ratio > 0.13` for at least `2.0 s`
- `looking_up`: `pitch_ratio < -0.10` for at least `2.0 s`
- `looking_side`: `gaze_ratio < -0.13` or `> 0.13` for at least `1.5 s`
- `partial_face`: face width `< 80 px` or face height `< 95 px`, then active after duration gate and scored after `5 s`
- `fake_presence`: low variance plus no blink
- `speaker_audio`: speech active while face is absent or lips are not speaking, held for `0.3 s`
- `phone`, `book`, `headphone`, `earbud`: object-vote thresholds inside a `15`-frame window
- `multiple_people`: person count `> 1`
- `no_person`: person count `== 0` and no landmark-derived motion signal

The `face_hidden` and `no_person` logic deserves special attention. In [main.py](/home/krushang/Desktop/ai-proctor-vision-poc/main.py#L388), these conditions are approximated using:

- `people_count > 0 and not (yaw or pitch or gaze)` for `face_hidden`
- `people_count == 0 and not (yaw or pitch or gaze)` for `no_person`

Because `yaw`, `pitch`, and `gaze` are floating-point values rather than explicit face-detected flags, this is a heuristic proxy rather than a clean face-presence test. It works reliably only when missing landmarks cause all three values to be zero.

## 8. Performance Considerations Across the Whole System

The main runtime bottlenecks are:

- YOLO inference on every frame.
- Two separate Face Mesh inferences on every frame.
- Synchronous drawing and UI rendering.

The current implementation is appropriate for a proof-of-concept, but large-scale or high-FPS deployment would likely require:

- shared landmark inference
- GPU-aware model scheduling
- optional frame skipping
- explicit asynchronous separation of vision and UI tasks

## 9. Global Limitations of the Current Implementation

- The system is highly heuristic and threshold-driven.
- Face-related computations rely on a single face even when multiple people are present.
- There is no central uncertainty modeling or confidence propagation.
- The object detector threshold for `person` is configured inconsistently.
- Some semantic conditions are inferred indirectly, especially `face_hidden` and `no_person`.
- The lip and head modules redundantly run Face Mesh.

## 10. High-Value Improvements for a Future Version

- Fuse face, gaze, blink, and lip analysis into one shared landmark pipeline.
- Introduce proper 3D head-pose estimation.
- Replace heuristic fake-presence detection with a dedicated anti-spoofing model.
- Add track-based person counting and object tracking.
- Use calibrated per-module confidence estimates and temporal probabilistic fusion.
- Separate policy logic from detection logic more formally for easier academic evaluation and ablation studies.
