# Multi-User Scaling Design — AI Proctor

> **Status:** Pre-implementation research document.
> **Purpose:** Complete record of the scaling discussion, architectural analysis, constraints,
> options considered, and the agreed implementation path before any code is written.
> **Next step:** Clone the repo, start implementation in the new folder using this document as the spec.

---

## 1. The Question

The current system (`main.py`) proctors **one candidate at a time** using a local webcam. The goal is to
switch the video/audio input to **WebRTC** (so candidates stream from their own devices) and run
inference on a **central server**, proctoring multiple candidates simultaneously from a single running
instance of the model.

The specific question: **how many concurrent users can a single instance support, and what needs to
change to get there?**

Hardware constraints were explicitly excluded from this discussion. The focus is purely on
**software and architectural constraints**.

---

## 2. Current Architecture (Single-User)

### 2.1 Processing Loop

`main.py` runs a single sequential loop:

```
while True:
    frame = cap.read()              # one webcam
    YOLO(frame)                     # object detection
    MediaPipe FaceMesh(frame)       # head/gaze/blink
    LipDetector(frame)              # MAR analysis
    AudioMonitor.speech_active()    # silero-VAD (separate thread)
    → RiskEngine.process_event()    # scoring
    → AlertEngine.handle()          # warn/alert routing
    → ProofWriter.push_frame()      # evidence capture
    cv2.imshow()                    # display
```

Everything happens sequentially in one thread for one user.

### 2.2 Complete Component Inventory

| Component | File | Role |
|---|---|---|
| `ObjectDetector` | `detectors/object_detector.py` | YOLO — phone, book, headphone, earbud, person |
| `HeadPoseDetector` | `detectors/head_pose_detector.py` | MediaPipe FaceMesh — yaw, pitch, gaze, blink, EAR |
| `LipDetector` | `detectors/lip_detector.py` | MAR + variance + oscillation → is_speaking |
| `AudioMonitor` | `core/audio_monitor.py` | pyaudio + silero-VAD → speech_active(); 30s ring buffer |
| `SpeakerAudioDetector` | `core/audio_monitor.py` | speech_active + lip_speaking → desync flag |
| `ObjectTemporalTracker` | `core/object_tracker.py` | rolling vote window (15 frames, per-class) |
| `HeadTracker` | `core/head_tracker.py` | duration gate — condition must persist N seconds |
| `LivenessDetector` | `core/liveness.py` | weighted variance + blink timeout → fake presence |
| `RiskEngine` | `core/risk_engine.py` | two-bucket scoring, state machine, timers, combos |
| `AlertEngine` | `core/alert_engine.py` | warn vs alert routing, cooldowns |
| `ProofWriter` | `utils/proof_writer.py` | image/video/AV evidence capture |

---

## 3. State Isolation Analysis

The first question before thinking about concurrency: which components hold per-user state,
and which could theoretically be shared?

### 3.1 Components that CANNOT be shared (hold per-user state)

| Component | Per-User State Held |
|---|---|
| `HeadPoseDetector` | `ear_buffer`, `blink_count`, `blink_frames`, `prev_ear` — all per-candidate history |
| `LipDetector` | `_state`: MAR history deque, velocity window, hold counter, yawn timer |
| `AudioMonitor` | `_audio_ring` (timestamped PCM ring), `_speech_detected`, pyaudio stream |
| `SpeakerAudioDetector` | `_no_lips_since`, `_flagged` |
| `ObjectTemporalTracker` | `_windows` dict — rolling vote deques per object class |
| `HeadTracker` | `states` dict — `start_time` per condition key |
| `LivenessDetector` | `_yaw_samples`, `_gaze_samples`, `_pitch_samples`, blink history, `_last_sample_time` |
| `RiskEngine` | scores, occurrences, all timers, state machine, combo cooldowns, decay log |
| `AlertEngine` | `_api_cooldown_until`, `_warn_cooldown_until`, `_termination_alerted` |
| `ProofWriter` | `_frames` deque (150-frame rolling buffer) |

**Every component that holds any history or timer is per-user. None of these can be shared.**

### 3.2 Components that CAN be shared

| Component | Share Condition |
|---|---|
| `ObjectDetector` (YOLO weights) | Yes, **with a mutex** for thread safety — or better: **batch inference** |
| `settings/scoring.py` values | Yes, read-only constants |
| `settings/alerts.py` values | Yes, read-only constants |
| Proof directory on disk | Yes, as long as per-user session subdirectories are used |

**The YOLO model is the only meaningful shared resource.**

---

## 4. Why Sequential Multi-User Breaks

### 4.1 The Naive Approach (and why it fails)

The obvious first attempt: run N users round-robin in the same loop:

```
while True:
    for user in users:
        frame = user.get_frame()
        process(frame, user.state)
```

This simply divides the available fps by N. At 10fps system throughput with 5 users: **2fps per user**.

### 4.2 The Temporal Assumption Problem

This is the **architectural** bottleneck — not a hardware question.

Several components assume a consistent, reasonably high frame rate. When fps drops, they break:

#### ObjectTemporalTracker

```python
OBJECT_WINDOW    = 15   # frames
BOOK_MIN_VOTES   = 10   # must appear in 10 of last 15 frames
```

The window is frame-count-based. At different effective fps:

| Effective fps | 15-frame window = | Meaning |
|---|---|---|
| 30fps | 0.5 seconds | Object must be consistently present for 0.5s ✓ |
| 10fps | 1.5 seconds | Slightly longer, still reasonable ✓ |
| 2fps | 7.5 seconds | Object from 7 seconds ago still influences current decision ✗ |
| 1fps | 15 seconds | Completely broken — "temporal stability" means nothing ✗ |

At 5 users sequential, every object detection — phone, book, earbud — starts behaving
unexpectedly. False positives take 7+ seconds to clear. A brief appearance can trigger a vote
that stays relevant for 7 seconds. The whole stability system is meaningless.

#### LivenessDetector

```python
SAMPLE_INTERVAL  = 0.2    # seconds — intended sampling rate
FAKE_WINDOW      = 15.0   # seconds of samples used for variance
NO_BLINK_TIMEOUT = 10     # seconds without blink → fake presence
```

At 2fps, you sample yaw/gaze/pitch every 0.5 seconds instead of every 0.2 seconds.
The variance window (`FAKE_WINDOW = 15s`) ends up with 30 samples instead of 75.
A real candidate who happens to be sampled during still moments (e.g., reading) looks
statistically fake. False positive rate increases significantly.

#### HeadTracker and RiskEngine timers

These use `time.time()` (wall clock) — **they are fine**. Duration gates (`LOOKING_AWAY_THRESHOLD = 1.5s`)
fire based on real seconds, not frame count. These components scale correctly.

#### Audio (SpeakerAudioDetector, AudioMonitor, RiskEngine speaker timer)

Runs in a **completely independent thread** per user. Zero dependency on video frame rate.
**Audio scales perfectly to N users with no changes** — each user just gets their own
`AudioMonitor` thread and `SpeakerAudioDetector` instance.

---

## 5. Options Considered

### Option A: One Process Per User (Multi-Process)

**How it works:**
Spin up a separate Python process for each candidate. Each process loads all models
independently and processes one user.

**Pros:**
- Complete isolation — zero shared state, zero concurrency issues
- No code changes to core logic whatsoever
- Crash in one user's process doesn't affect others
- MediaPipe, YOLO, everything fully independent

**Cons:**
- YOLO model loaded N times → N × GPU/CPU memory for weights (~300MB–1GB per model)
- No YOLO batching benefit — N separate forward passes
- Spawning processes per WebRTC connection adds overhead
- At 10 users: 10 × YOLO = significant memory pressure

**Verdict:** Simplest to implement. Works well for 2–3 users. Not efficient for 5–10.

---

### Option B: One Thread Per User, Sequential YOLO with Lock

**How it works:**
N threads, each handling one user's full pipeline. YOLO is shared with a threading lock —
only one thread calls `model(frame)` at a time.

```python
with yolo_lock:
    detections = model(frame)
```

**Pros:**
- Single YOLO model in memory
- MediaPipe per-thread (separate instances, genuine parallelism via GIL release)
- Simple to implement

**Cons:**
- YOLO calls serialize → same N× latency problem as sequential processing
- The lock contention makes it no better than the naive approach for vision throughput
- Audio threads are fine but vision bottlenecks at YOLO

**Verdict:** Doesn't solve the core problem. Not chosen.

---

### Option C: YOLO Batching + Per-User MediaPipe Threads ✓ CHOSEN

**How it works:**

```
Every tick:
1. Collect current frames from all N WebRTC streams
2. Single YOLO batch call: model([f1, f2, ..., fN]) → [dets1, dets2, ..., detsN]
3. Thread pool: N threads each run their own MediaPipe instance on their frame (parallel)
4. Per-user: update state machines with results (fast, CPU-light)
5. Per-user: audio state read from independent audio threads (no coordination needed)
```

**Why YOLO batching works:**
Ultralytics supports batch inference natively:
```python
results = model([frame1, frame2, frame3, frame4, frame5])
# Single GPU forward pass. Returns list of Results, one per input frame.
# Latency ≈ single-frame latency (batch overhead is small).
```

**Why MediaPipe parallelism works:**
Each `mp.solutions.face_mesh.FaceMesh()` is an independent C++ object.
When `.process(image)` is called, the heavy computation runs in C++ which
**releases the Python GIL** — the same mechanism that allows NumPy, PyTorch, and
OpenCV operations to run truly in parallel across threads.

```python
# This is genuinely parallel — GIL released during C++ computation
Thread 1: user1_face_mesh.process(frame1)
Thread 2: user2_face_mesh.process(frame2)
Thread 3: user3_face_mesh.process(frame3)
```

The Python overhead (result parsing, array creation) is small and serialised, but the
bulk of the computation runs concurrently.

**Why audio requires no changes:**
Each user's `AudioMonitor` already runs in a daemon thread pulling from their WebRTC
audio stream. The `speech_active()` call from the main loop just reads a lock-protected
boolean. This is already the correct design.

---

### Option D: Time-Based ObjectTemporalTracker

Not an independent option but a **necessary companion fix** for any approach at 5–10 users.

**Current design:**
```python
OBJECT_WINDOW    = 15   # last N frames
BOOK_MIN_VOTES   = 10   # must appear in N frames
```

**Proposed design:**
Track `(timestamp, present)` pairs. Instead of "was present in 10 of last 15 frames",
the check becomes "was present in at least 10 observations within the last 1.5 seconds".

```python
OBJECT_WINDOW_S      = 1.5    # time window in seconds
BOOK_MIN_VOTES       = 10     # must appear in N observations within the window
```

This makes the tracker's behaviour consistent at any frame rate:
- At 30fps: 45 observations in 1.5s, need 10 → easy to trigger, fast to clear
- At 5fps: 7–8 observations in 1.5s, need 10 → slightly harder, might need
  window expansion or vote reduction
- The **semantics stay the same** regardless of fps

This is the fix that pushes the 10-user boundary from "broken" to "workable".

---

## 6. Chosen Architecture: Option C + Option D

### 6.1 Component Model

```
┌──────────────────────────────────────────────────────────────────────┐
│  ProctorCoordinator  (new — replaces main.py)                        │
│                                                                      │
│  Shared:                                                             │
│  ┌─────────────────────────────────────────────────────┐            │
│  │  ObjectDetector  (single YOLO model, batch inference) │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                      │
│  Per-user  (ProctorSession):                                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  HeadPoseDetector  (MediaPipe FaceMesh instance)               │  │
│  │  LipDetector                                                   │  │
│  │  ObjectTemporalTracker  (time-based window)                    │  │
│  │  HeadTracker                                                   │  │
│  │  LivenessDetector                                              │  │
│  │  AudioMonitor  (WebRTC audio stream → silero-VAD thread)       │  │
│  │  SpeakerAudioDetector                                          │  │
│  │  RiskEngine                                                    │  │
│  │  AlertEngine                                                   │  │
│  │  ProofWriter                                                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Main Loop Pseudocode

```python
class ProctorCoordinator:
    def __init__(self, max_users):
        self.detector   = ObjectDetector()   # single shared YOLO
        self.sessions: dict[str, ProctorSession] = {}
        self.mp_pool    = ThreadPoolExecutor(max_workers=max_users)

    def tick(self):
        # 1. Collect frames from all active sessions
        user_ids = list(self.sessions.keys())
        frames   = [self.sessions[uid].get_latest_frame() for uid in user_ids]

        # 2. Single YOLO batch call
        if frames:
            batch_detections = self.detector.detect_batch(frames)
        else:
            return

        # 3. Parallel MediaPipe (thread pool — GIL released during C++ compute)
        futures = {
            uid: self.mp_pool.submit(self.sessions[uid].run_mediapipe, frame)
            for uid, frame in zip(user_ids, frames)
        }
        mp_results = {uid: f.result() for uid, f in futures.items()}

        # 4. Per-user state update (fast, sequential fine at this stage)
        for i, uid in enumerate(user_ids):
            self.sessions[uid].update(
                detections = batch_detections[i],
                mp_result  = mp_results[uid],
                frame      = frames[i],
            )
```

### 6.3 ProctorSession Responsibilities

`ProctorSession` is a new class that encapsulates everything currently at the top of `main()`:

```python
class ProctorSession:
    def __init__(self, session_id, session_dir):
        self.session_id      = session_id
        self.head_detector   = HeadPoseDetector()
        self.lip_detector    = LipDetector()
        self.obj_tracker     = ObjectTemporalTracker(...)   # time-based
        self.head_tracker    = HeadTracker(...)
        self.liveness        = LivenessDetector(...)
        self.audio_monitor   = AudioMonitor(...)            # WebRTC audio stream
        self.speaker_audio   = SpeakerAudioDetector(...)
        self.risk            = RiskEngine(...)
        self.alert_engine    = AlertEngine()
        self.proof_writer    = ProofWriter(session_dir / "proof")
        self.alert_log       = []
        self.warning_log     = []
        # ...

    def run_mediapipe(self, frame):
        """Called from thread pool — runs HeadPoseDetector and LipDetector."""
        return self.head_detector.detect(frame, draw=False), \
               self.lip_detector.process(frame, draw=False)

    def update(self, detections, mp_result, frame):
        """Per-user state machine update — runs after YOLO + MediaPipe."""
        head_result, lip_result = mp_result
        # ... same logic as current main loop, but scoped to this session
```

---

## 7. Scaling Estimates

With Option C (YOLO batching + parallel MediaPipe):

| Users | YOLO cost | MediaPipe (parallel) | Effective fps/user | ObjectTracker (1.5s window) | Status |
|---|---|---|---|---|---|
| 1 | 1× batch | 1 thread | ~10–15fps | ✓ 15–22 obs/window | ✓ Full fidelity |
| 2 | 1× batch | 2 threads | ~9–12fps | ✓ 13–18 obs/window | ✓ |
| 5 | 1× batch (~1.3×) | 5 threads | ~6–9fps | ✓ 9–13 obs/window | ✓ Good |
| 10 | 1× batch (~2×) | 10 threads | ~4–6fps | ⚠️ 6–9 obs/window | ⚠️ Borderline |

At 10 users, `BOOK_MIN_VOTES = 10` at 4fps with a 1.5s window gives only 6 observations —
you'd need to reduce `MIN_VOTES` or extend the window to 2.5s for the 10-user case.

The audio path is **unaffected at any user count**.

---

## 8. Changes Required in the Codebase

This section maps every concrete code change needed. None of these touch the
existing single-user logic — they are new classes/methods added on top.

### 8.1 `detectors/object_detector.py`

Add `detect_batch(frames: list) -> list[list[dict]]`:

```python
def detect_batch(self, frames: list) -> list[list[dict]]:
    """Run YOLO on a batch of frames. Returns one detection list per frame."""
    all_results = self.cheat_model(frames, verbose=False)
    return [self._parse_results([r]) for r in all_results]
```

The single-user `detect(frame)` method is unchanged — it stays for backward compatibility.

### 8.2 `core/object_tracker.py`

Convert `ObjectTemporalTracker` from frame-count-based to time-based window:

```python
# Current:
OBJECT_WINDOW = 15    # last N frames

# New:
OBJECT_WINDOW_S = 1.5   # last N seconds
```

Internal change: store `deque[(timestamp, present)]` instead of `deque[bool]`.
Vote logic: count `present=True` entries where `time.time() - timestamp <= window_s`.

The `per_key_min_votes` dict stays the same — only the window type changes.

### 8.3 New: `core/proctor_session.py`

New file. Wraps all per-user state into one class.
This is basically the current `main()` body lifted into a class with `__init__` and `update()`.
No logic changes — just re-scoped.

### 8.4 New: `core/proctor_coordinator.py`

New file. Manages the collection of `ProctorSession` instances and the shared `ObjectDetector`.
Runs the tick loop: collect frames → batch YOLO → parallel MediaPipe → per-user update.

### 8.5 `core/audio_monitor.py`

The `AudioMonitor` currently opens a `pyaudio` stream from the local mic.
For WebRTC, the audio input changes — instead of `pyaudio.open(input=True)`,
audio chunks arrive via WebRTC callbacks.

The internal ring buffer and `get_audio_range()` logic is **unchanged**.
Only the data source changes: replace `pyaudio` read loop with a WebRTC audio callback
that calls `_audio_ring.append((time.time(), chunk_bytes))`.

### 8.6 WebRTC Integration Layer (new)

Not part of the existing codebase — this is a new layer on top.
Options: `aiortc` (Python WebRTC), or a media server (mediasoup, Janus) that proxies
frames/audio into the Python coordinator.

The coordinator doesn't care about the transport — it just needs:
- `get_latest_frame() -> np.ndarray` per user
- `push_audio_chunk(bytes, timestamp)` per user

---

## 9. What Does NOT Need to Change

This is equally important — these components are correct as designed:

| Component | Reason no change needed |
|---|---|
| `RiskEngine` | Fully stateless relative to other users. All timers are wall-clock. |
| `AlertEngine` | Per-session cooldown tables. No shared state. |
| `HeadTracker` | Wall-clock duration gates. Works at any fps. |
| `LivenessDetector` | Wall-clock sampling. Minor degradation at very low fps but logic is sound. |
| `SpeakerAudioDetector` | Audio thread independent. Unaffected. |
| `ProofWriter` | Per-session frame buffer. No shared state. |
| `settings/scoring.py` | Read-only constants. |
| `settings/alerts.py` | Read-only constants. |
| All score values, thresholds, cooldowns | Nothing changes for multi-user. |

---

## 10. Open Questions for Implementation

1. **WebRTC library**: `aiortc` vs dedicated media server (mediasoup/Janus) feeding frames into Python?
   - `aiortc` keeps everything in Python, simpler stack
   - Media server offloads transport, lets Python focus on inference
   - Decision affects how `AudioMonitor` receives chunks

2. **Coordinator tick rate**: Should the coordinator run at a fixed tick rate (e.g., `asyncio` loop at 15fps)
   or free-running (process as fast as possible)?
   - Fixed tick: predictable timing for all users
   - Free-running: maximises throughput but harder to reason about timing

3. **MediaPipe thread pool size**: Should it equal `max_users` or be smaller?
   - Equal: maximum parallelism
   - Smaller: avoids thread overhead for small user counts, pool reuses threads

4. **Session lifecycle**: How are sessions created/destroyed as candidates connect/disconnect?
   Needs a clean `start_session(user_id)` / `end_session(user_id)` API with report saving on disconnect.

5. **ObjectTemporalTracker window tuning for variable fps**: Should the window be fixed at 1.5s
   for all user counts, or should it be configurable based on expected fps
   (e.g., `OBJECT_WINDOW_S = 1.5` at ≤5 users, `2.5` at ≤10 users)?

6. **Proof video fps**: `_actual_fps()` already derives fps from timestamps — this is already
   correct for multi-user (each user's proof video will reflect their actual capture rate). No change needed.

---

## 11. Summary

| Aspect | Current | Target |
|---|---|---|
| Users | 1 | 5–10 |
| YOLO | 1 frame/call | N frames/call (batch) |
| MediaPipe | 1 instance, main thread | N instances, thread pool |
| Audio | 1 pyaudio thread | N WebRTC audio callbacks |
| State management | Top-level vars in `main()` | `ProctorSession` per user |
| `ObjectTemporalTracker` | Frame-count window | Time-based window |
| All scoring/detection logic | Unchanged | Unchanged |
| All settings/thresholds | Unchanged | Unchanged |

The core insight: **the detection and scoring logic is already correct for multi-user**.
The only things that need changing are:
1. How frames reach the models (WebRTC instead of local webcam)
2. How models are called (batched YOLO, pooled MediaPipe)
3. How state is organised (ProctorSession class instead of `main()` variables)
4. The ObjectTemporalTracker window type (time-based instead of frame-count-based)

The risk scoring system, alert routing, proof capture, session reports — all of it transfers
to multi-user without a single line of logic change.
