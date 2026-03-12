# AI Proctoring Risk Scoring System Design

> **Document status:** Updated — reflects all design changes made after initial implementation.
> Each section that changed includes a **▶ Change Log** block explaining what was changed, what it was before, and why.

---

## 1. Purpose of This Document

This document defines the complete scoring and risk evaluation system used by the AI proctoring platform. The purpose of the scoring system is to:

1. Detect cheating behavior reliably.
2. Avoid false positives that may penalize honest candidates.
3. Provide transparent reasoning for every decision.
4. Allow administrators to review suspicious activity.
5. Terminate exams only when strong evidence of cheating exists.

The system is intentionally designed as a **risk accumulation engine rather than a single-event trigger system**. This ensures that no candidate is penalized due to momentary detection errors.

All tunable values (scores, cooldowns, thresholds) live in `settings/scoring.py` and `settings/alerts.py`. Engine logic files (`core/risk_engine.py`, `core/alert_engine.py`) contain no hardcoded numbers.

---

## 2. Core Design Philosophy

### 2.1 No Single Frame Decisions

Computer vision models occasionally produce noisy predictions. Therefore:

- No decision is made from a single frame.
- All events must persist for a defined duration before triggering.
- Object detections use a sliding-window vote system (ObjectTemporalTracker).

This prevents false positives caused by lighting changes, momentary head movement, and detection glitches.

---

### 2.2 Warning vs Alert — Strict Separation

> **▶ Change Log**
> **Before:** The system had a `WARN_FIRST_N` dict in AlertEngine that forced the first N occurrences to show a warning regardless of whether score was added. This created a misleading state where score could be added silently while a warning was shown on screen.
>
> **After:** The rule is now absolute and defined in AlertEngine:
> - `risk_added == 0` → **WARNING** (amber, on-screen only, never logged)
> - `risk_added  > 0` → **ALERT** (red, logged to report, proof saved)
>
> **Why:** A warning must honestly mean "no score was added." If score is going up, the invigilator must see a red alert, not a soft warning. The old system was deceptive for gaze events where `WARN_FIRST_N=1` suppressed the alert but the score already moved.

**Warning (amber):**
- Shown when no score was added.
- Covers: grace period, active score cooldown, duration gate not yet met.
- Displayed on-screen only. Not logged to report.

**Alert (red):**
- Shown whenever score is actually added.
- Logged to session report with timestamp, score added, and proof file path.
- Message includes score added: e.g. `"Earbuds detected  [+20 pts]"`

---

### 2.3 Grace Period — Enforced in RiskEngine, Not AlertEngine

> **▶ Change Log**
> **Before:** Grace was controlled by `WARN_FIRST_N` in AlertEngine. AlertEngine checked the occurrence count and forced a warning display. RiskEngine still scored some events (gaze) during grace, creating inconsistency.
>
> **After:** Grace is enforced entirely in RiskEngine. For occurrence-based events (phone, headphone, earbud, multiple_people), `occ=1` calls `_arm_cooldown()` without adding any score. AlertEngine receives `risk_added=0` and naturally routes to warning.
>
> **Why:** Grace should mean the engine genuinely did not add score — not just that AlertEngine hid the alert. This makes the system consistent: if you see a warning, zero score was added. Period.

**Grace also arms the score cooldown (same duration):**

> **▶ Change Log**
> **Before:** occ=1 had no cooldown. The second person could enter frame (occ=1, warning), leave for 1 second, and re-enter (occ=2, immediate alert). No delay.
>
> **After:** occ=1 arms the score cooldown. occ=2 cannot fire an alert until the cooldown expires.
>
> **Why:** Without this, grace is trivially exploited by briefly leaving and re-entering frame. Now the same wait time applies whether you are in grace or in cooldown.

---

### 2.4 Three Cooldown Types and Their Rules

> **▶ Change Log**
> **Before:** The three cooldowns (score, warn, API alert) were set independently with no enforced relationship. This led to warn_cooldown > score_cooldown (warning showed less often than scoring happened) and api_cooldown >> score_cooldown (score accumulated silently for long periods with no on-screen feedback).
>
> **After:** Two strict rules are enforced:
> 1. `api_cooldown == score_cooldown` — every scoring event directly fires an alert.
> 2. `warn_cooldown <= score_cooldown` — warning shows at least as often as scoring.
>
> **Why:** Score climbing silently while the invigilator sees nothing is wrong. If score goes up, there should be an alert. And a warning should never lag behind the score.

| Cooldown | File | Rule | Purpose |
|---|---|---|---|
| `SCORE_COOLDOWNS` | `settings/scoring.py` | Source of truth | Minimum gap between score additions |
| `API_COOLDOWNS` | `settings/alerts.py` | Must equal `SCORE_COOLDOWNS` | Every score fires a logged alert |
| `WARN_COOLDOWNS` | `settings/alerts.py` | Must be ≤ `SCORE_COOLDOWNS` | Warning shown during score cooldown gap |

---

### 2.5 Condition Categories

> **▶ Change Log**
> **Before:** All conditions were treated uniformly with the same occurrence-based grace mechanism.
>
> **After:** Conditions are divided into four categories with different grace mechanisms.

#### Category A — Occurrence-based
`phone`, `earbud`, `headphone`, `book`

These are objects that appear and disappear. Each detection episode is one occurrence.

```
occ=1  →  WARNING + cooldown armed (grace, no score)
occ=2+ →  ALERT + score (if cooldown clear)
           score cooldown active →  WARNING
```

#### Category B — Time-based
`looking_away`, `looking_down`, `looking_up`, `looking_side`, `partial_face`

Continuous head/gaze conditions. Grace is the HeadTracker duration gate (1.5s), not occurrence count.

```
HeadTracker gate (1.5s) = natural grace, nothing shown during gate
Condition triggers → ALERT + score immediately
Every score_cooldown seconds while still active → ALERT again
During score cooldown → WARNING
```

> **▶ Change Log — Gaze score cooldown**
> **Before:** `SCORE_COOLDOWNS["looking_*"] = 3s`. Score fired every 3 seconds.
> **After:** `SCORE_COOLDOWNS["looking_*"] = 5s`.
> **Why:** 3s was too aggressive — a 2-minute session of looking away would generate 40 alert log entries. 5s is still frequent enough to build score quickly while keeping the report readable.

#### Category C — Duration-tiered
`face_hidden`, `fake_presence`

Duration gate determines the warning zone. Score increases with time in tiers.

```
Duration < tier-1 threshold  →  WARNING  (duration gate not met, risk_added=0)
Duration ≥ tier-1            →  ALERT + tier-1 score
Duration ≥ tier-2            →  ALERT + tier-2 score (larger)
```

> **▶ Change Log — Fake presence tiers**
> **Before:** `<15s = warning, ≥15s = +30, ≥40s = +60`
> **After:** `<10s = warning, ≥10s = +30, ≥25s = +60`
> **Why:** 15s was too long before any penalty. 10s is still generous for accidental stillness while being fast enough to catch a static image.

#### Category D — Hybrid (count + time)
`multiple_people`, `no_person`

Tracked both by occurrence count (grace, occ=1 warning) and by continuous duration (tiers, termination at 20s). Handled in `_handle_special` in RiskEngine.

```
multiple_people:
  occ=1          → WARNING + cooldown armed
  occ=2          → ALERT + score (+20)
  occ≥3          → ALERT + score (+50)
  continuous >20s → TERMINATE

no_person:
  duration ≥ 5s  → ALERT + score (+25)
  duration ≥ 10s → ALERT + score (+50)
  duration ≥ 20s → TERMINATE
```

> **▶ Change Log — no_person risk_added**
> **Before:** `_handle_special` for no_person returned `risk_added=0.0` in the RiskEvent even when score was added internally. AlertEngine always saw 0 and showed a warning.
> **After:** `_handle_special` tracks the actual score added and returns it in `risk_added`.
> **Why:** Score was silently added while the invigilator saw a warning — violating the `risk_added>0 = alert` rule.

#### Category E — Duration-tiered with continuous timer
`speaker_audio`

Speech detected without matching lip movement. Uses an independent continuous timer that resets only on sustained silence (flicker grace applied). Tier gates survive brief silence episodes to prevent gaming.

```
0s – 3s (WARN_DURATION)   →  WARNING every WARN_COOLDOWN (3s)
at 3s                      →  ALERT + score (+10 fixed, one-time per episode)
at 13s                     →  ALERT + score (+25 fixed, one-time per episode)
every 10s after 13s        →  ALERT + score (+15 fixed, repeating)
silence > flicker_grace_s  →  timer resets; tier gates PRESERVED
```

---

### 2.6 Progressive Escalation

Instead of large immediate penalties the system escalates gradually:

1. Warning (no score, grace or cooldown)
2. Alert with minor score (first scoring event)
3. Repeated alerts with accumulating score
4. Admin Review state (score ≥ 100)
5. Termination (continuous rules or score threshold)

---

### 2.7 Risk Score Instead of Immediate Termination

All suspicious activities accumulate into a **risk score**. Termination via score only occurs at Admin Review level (score ≥ 100). This ensures multiple suspicious behaviors are required before action.

---

## 3. Audio Proctoring — Removed and Re-added

### 3.1 What Was Removed (Old Audio System)

> **▶ Change Log**
> **Before:** The system included a full speaker-verification pipeline:
> - `core/audio_proctoring/` — ProctorSession, CheatType, CheatEvent (resemblyzer embeddings)
> - Enrollment via `enrollment.wav` at startup
> - Noise profiling + denoising via `noisereduce`
> - Audio streamed via `sounddevice` InputStream
> - Speaker diarization: classified audio as candidate voice vs external speaker
>
> **After:** Entire old pipeline removed.
>
> **Why:** Audio analysis produced unreliable results in typical noisy environments (measured SNR ~6.5 dB; resemblyzer needs 15+ dB). Speaker separation was weak (silhouette=0.321). The enrollment requirement added friction. The complexity (ProctorSession, threading, queue, denoising, embeddings) was disproportionate to reliability.

**Removed components:**
- `core/audio_proctoring/` — ProctorSession, CheatType, CheatEvent
- `sounddevice`, `soundfile`, `noisereduce`, `resemblyzer` dependencies
- Enrollment WAV loading at startup
- Noise profiling from first 0.5s of mic audio

### 3.2 What Was Added (New Audio System)

> **▶ Change Log**
> **New approach:** Detect speech from an external device (phone speaker, laptop speaker) by checking for **audio-lip desync** — microphone picks up speech while the candidate's lips are not moving.
>
> **Why:** Simpler, more reliable, and no enrollment needed. A candidate speaking naturally will have lip movement; audio played from a device will not. The goal is not to identify who is speaking but whether the speech is coming from the candidate's mouth.

**New components:**
- `core/audio_monitor.py` — `AudioMonitor`: streams microphone via `pyaudio`, runs silero-VAD per chunk, exposes `speech_active()`. Stores a 30s timestamped ring buffer for proof audio extraction.
- `core/audio_monitor.py` — `SpeakerAudioDetector`: combines `speech_active` + `lip_speaking` + `face_detected` to flag desync. Holds the flag for `SPEAKER_HOLD_S` after signal appears.
- `detectors/lip_detector.py` — `LipDetector`: MAR (Mouth Aspect Ratio) + dynamic variance + oscillation analysis. Outputs `is_speaking`, `is_yawning`, `face_detected`.
- `core/risk_engine.py` — `_handle_special("speaker_audio")`: duration-tiered scoring with tier gates that survive silence resets.

---

## 4. Detailed Event Policies (Current)

### 4.1 Phone Detection

**Category:** Occurrence-based

| Occurrence | Action | Score |
|---|---|---|
| occ = 1 | WARNING — grace, cooldown armed | 0 |
| occ = 2 | ALERT | +25 × confidence (non-decaying) |
| occ ≥ 3 | ALERT | +50 × confidence (non-decaying) |

- Score cooldown: 15s (also arms on occ=1 grace)
- YOLO confidence threshold: 0.65 (`YOLO_PHONE_CONF`)
- Vote threshold: 9/15 frames (`PHONE_MIN_VOTES`)

**Combo bonus:** phone + looking_down simultaneously → +20 (decaying, 60s internal cooldown)

> **▶ Change Log:** Score cooldown was 15s before and remains 15s. API cooldown reduced from 60s to 15s (now equals score_cooldown). Phone vote threshold set to 9/15 frames to reduce brand-text false positives.

---

### 4.2 Book Detection

**Category:** Occurrence-based (no grace — immediate scoring)

| State | Action | Score |
|---|---|---|
| First stable detection | ALERT | +20 × confidence (decaying) |
| Every 30s while active | ALERT | +20 × confidence (decaying) |
| Score cooldown active | WARNING | 0 |

- Score cooldown: 30s
- YOLO confidence threshold: 0.70 (`YOLO_BOOK_CONF`)
- Vote threshold: 10/15 frames (`BOOK_MIN_VOTES`)

**Combo bonus:** looking_down + book simultaneously → +15 (decaying, 60s internal cooldown)

> **▶ Change Log — vote threshold and confidence**
> **Before:** Vote threshold was 5/15 (same as default). YOLO confidence threshold was 0.65.
> **After:** Vote threshold raised to 10/15. YOLO confidence threshold raised to 0.70. Both thresholds moved to `config.py` as `BOOK_MIN_VOTES` and `YOLO_BOOK_CONF`.
> **Why:** Book detection had a high false-positive rate — common objects (notebooks, paper, desk surfaces) were misclassified. Raising both the vote count and the YOLO confidence gate significantly reduces spurious detections without missing real books.

---

### 4.3 Headphone Detection

**Category:** Occurrence-based

| Occurrence | Action | Score |
|---|---|---|
| occ = 1 | WARNING — grace, cooldown armed | 0 |
| occ ≥ 2 | ALERT | +20 × confidence (decaying) |

- Score cooldown: 30s
- YOLO confidence threshold: 0.41 (`YOLO_AUDIO_CONF`)
- Vote threshold: 5/15 frames (default)

> **▶ Change Log:** Score was +10, raised to +20. Score cooldown was 120s, reduced to 30s. API cooldown was 180s, now 30s. Grace cooldown now armed on occ=1 (was not armed before).

---

### 4.4 Earbud Detection

**Category:** Occurrence-based

| Occurrence | Action | Score |
|---|---|---|
| occ = 1 | WARNING — grace, cooldown armed | 0 |
| occ ≥ 2 | ALERT | +20 × confidence (decaying) |

- Score cooldown: 30s
- YOLO confidence threshold: 0.41 (`YOLO_AUDIO_CONF`)
- Vote threshold: 9/15 frames (`EARBUD_MIN_VOTES`)

> **▶ Change Log — vote threshold**
> **Before:** Vote threshold was 5/15 (same as default).
> **After:** Vote threshold raised to 9/15 (`EARBUD_MIN_VOTES` in `config.py`).
> **Why:** Earbud false-positive rate was high — hair accessories, ear shadows, and earrings were triggering detection. Higher vote count requires the object to be consistently visible across frames before scoring.

---

### 4.5 Multiple People Detection

**Category:** Hybrid (occurrence + continuous time)

| Occurrence | Action | Score |
|---|---|---|
| occ = 1 | WARNING — grace, cooldown armed | 0 |
| occ = 2 | ALERT | +20 × confidence (non-decaying) |
| occ ≥ 3 | ALERT | +50 × confidence (non-decaying) |
| continuous > 20s | TERMINATE | — |

- Score cooldown: 10s (also arms on occ=1 grace)
- Flicker grace: 1.5s (brief dropout does not reset continuous timer)
- **Proof:** image per alert occurrence; video (last 5s) on termination

> **▶ Change Log:** occ=2 score was +10, raised to +20. Grace cooldown now armed on occ=1 to prevent immediate re-exploit. API cooldown was 30s, now 10s (equals score_cooldown).

---

### 4.6 No Person Detected

**Category:** Hybrid (duration-tiered + termination)

| Duration | Action | Score |
|---|---|---|
| < 5s | WARNING | 0 |
| ≥ 5s | ALERT | +25 (non-decaying) |
| ≥ 10s | ALERT | +50 (non-decaying) |
| ≥ 20s | TERMINATE | — |

- Score cooldown: 10s per tier (internal per-tier cooldown: 15s)
- Flicker grace: 1.5s
- **Proof:** image per alert occurrence; video (last 5s) on termination

> **▶ Change Log:** RiskEvent previously returned `risk_added=0.0` even when score was added internally in `_handle_special`. AlertEngine always showed a warning. Fixed to return actual score added so alert fires correctly.

---

### 4.7 Gaze Direction Signals

**Category:** Time-based
**Includes:** `looking_away`, `looking_down`, `looking_up`, `looking_side`

| State | Action | Score |
|---|---|---|
| HeadTracker gate < 1.5s | Nothing shown | 0 |
| Gate cleared (occ=1) | ALERT | +5 × confidence (decaying) |
| Every 5s while persisting | ALERT | +5 × confidence (decaying) |
| During 5s score cooldown | WARNING | 0 |
| After gap >1.5s, re-triggers | ALERT (if cooldown clear) | +5 |

- Score cooldown: 5s (= API cooldown)
- Warn cooldown: 3s
- Gaze thresholds: left/right ±0.13 (GAZE_LEFT/GAZE_RIGHT in config.py)
- `looking_side` uses separate duration gate: 1.0s (faster trigger)

**Gaze aggregation bonus:** 3+ gaze events within 30 seconds → +10 (decaying)

> **▶ Change Log:** Score cooldown was 3s, raised to 5s. API cooldown was 45s, reduced to 5s (= score_cooldown). WARN_FIRST_N=1 removed — HeadTracker 1.5s gate serves as natural grace. Gaze threshold tightened from ±0.15 to ±0.13 (more sensitive). Aggregation trigger reduced from 5+ to 3+ events. looking_side threshold reduced from 1.5s to 1.0s.

---

### 4.8 Face Hidden

**Category:** Duration-tiered

| Duration | Action | Score |
|---|---|---|
| < 5s | WARNING | 0 |
| ≥ 5s | ALERT | +10 (decaying) |
| ≥ 10s | ALERT | +20 (decaying) |

- Score cooldown: 5s
- Condition: person detected by YOLO but face landmarks absent

> **▶ Change Log:** Score cooldown was 10s (separate from API cooldown of 30s). Now both 5s. WARN_FIRST_N=1 removed — duration gate naturally produces warning phase. Fixed double-counting: `face_hidden` previously fired when `people_count == 0` (same as no_person). Corrected to only fire when `people_count > 0 AND no face landmarks`.

---

### 4.9 Fake Presence (Static Image)

**Category:** Duration-tiered

| Duration | Action | Score |
|---|---|---|
| < 10s | WARNING | 0 |
| ≥ 10s | ALERT | +30 (non-decaying) |
| ≥ 25s | ALERT | +60 (non-decaying) |

- Score cooldown: 10s
- Detection: weighted variance across yaw (45%), gaze (45%), pitch (10%) + blink timeout

> **▶ Change Log:**
> **Before:** `<15s = warning, ≥15s = +30, ≥40s = +60`. Score cooldown was 15s, API cooldown 60s.
> **After:** `<10s = warning, ≥10s = +30, ≥25s = +60`. Both cooldowns now 10s.
> WARN_FIRST_N=1 removed — duration gate handles warning zone naturally.

---

### 4.10 Partial Face (Too Far from Camera)

**Category:** Time-based

| State | Action | Score |
|---|---|---|
| Face width < MIN_FACE_WIDTH OR height < MIN_FACE_HEIGHT | Banner shown | — |
| Active < 5s | WARNING | 0 |
| Active ≥ 5s | ALERT | +2 (decaying) |
| Every 5s while persisting | ALERT | +2 |

- Score cooldown: 5s
- Min face width: 80px, min face height: 95px (config.py)
- Visual: bold orange bottom banner "MOVE CLOSER TO CAMERA"

> **▶ Change Log:** Duration gate was 3s, raised to 5s to match score cooldown. API cooldown was 90s, reduced to 5s. Thresholds tuned from 120/140px down to 80/95px (allows candidate to sit at reasonable distance while still catching very small faces). Banner added as a prominent user-facing prompt.

---

### 4.11 Speaker Audio (Speech Without Lip Movement)

**Category:** Duration-tiered with continuous timer (Category E)

> **▶ Change Log — scoring approach**
> **Before (v1):** Speaker audio had a flat `SPEAKER_AUDIO_SCORE` entry in `SCORE_COOLDOWNS` and was handled by `_score_event` like any other key. A fixed score was added every cooldown interval.
>
> **After (v2):** Moved to `_handle_special` with a dedicated continuous timer and three escalating tier gates. All speaker audio knobs (`SPEAKER_WARN_DURATION`, `SPEAKER_WARN_COOLDOWN`, `SPEAKER_SCORE_1`, `SPEAKER_SCORE_2_AT`, `SPEAKER_SCORE_2`, `SPEAKER_SCORE_TAIL`, `SPEAKER_TAIL_INTERVAL`, `SPEAKER_ALERT_COOLDOWN`) are consolidated in a single block in `settings/scoring.py`.
>
> **Why:** Flat scoring treated a 1-second glitch the same as a 30-second sustained episode. Duration-tiered scoring correctly escalates: warn briefly, then score progressively. Consolidating all knobs in one block avoids them being split between `scoring.py` and `alerts.py`.

| Timeline | Action | Score |
|---|---|---|
| 0 – 3s | WARNING every 3s | 0 |
| at 3s | ALERT (tier 1, one-time per episode) | +10 fixed |
| at 13s | ALERT (tier 2, one-time per episode) | +25 fixed |
| every 10s after 13s | ALERT (tail, repeating) | +15 fixed |

**Silence reset behaviour:**

> **▶ Change Log — anti-gaming fix**
> **Before:** Silence reset cleared `_speaker_since` AND all three tier gate cooldowns (`_speaker_t1`, `_speaker_t2`, `_speaker_tail`). Stopping and restarting audio reset the full 3s grace on every re-entry.
>
> **After:** Silence (sustained beyond `flicker_grace_s`) only resets `_speaker_since` and `_speaker_gone_since`. Tier gates are **not cleared**.
>
> **Why:** Without this, a candidate could exploit the system by alternating 2-second bursts of speaker audio separated by brief pauses — each re-entry got the full 3s grace and tier-1 was never reached. Now re-entries after silence skip previously reached tiers and score sooner.

- Flicker grace: `TIMER_FLICKER_GRACE_S` (1.5s) — brief silence does not reset timer
- Score bucket: non-decaying (fixed)
- **Proof:** `.mp4` video + companion `.wav` audio file (or merged `.mkv` if ffmpeg available)

---

### 4.12 Exit Fullscreen

**Category:** Time-based (implemented but low priority)

| Occurrence | Action | Score |
|---|---|---|
| occ = 1 | WARNING | 0 |
| occ ≥ 2, duration ≥ 2s | ALERT | +5 (decaying) |

---

## 5. Confidence Weighting

Events only contribute risk if detection confidence meets the per-key threshold.

```
Risk += BaseScore × confidence
```

Per-key YOLO confidence thresholds — all in `config.py`:

| Key | Config constant | Value |
|---|---|---|
| default | `YOLO_DEFAULT_CONF` | 0.50 |
| person | `YOLO_PERSON_CONF` | 0.30 |
| phone | `YOLO_PHONE_CONF` | 0.65 |
| book | `YOLO_BOOK_CONF` | 0.70 |
| headphone + earbud | `YOLO_AUDIO_CONF` | 0.41 |

Head/gaze events always pass `confidence=1.0`.

> **▶ Change Log — centralised to config.py**
> **Before:** Confidence thresholds were hardcoded as default parameter values in `ObjectDetector.__init__()` in `detectors/object_detector.py`. To change a threshold you had to edit the detector source.
>
> **After:** All thresholds live in `config.py` under `# Object Detection (YOLO confidence thresholds)` and are passed explicitly when constructing `ObjectDetector()` in `main.py`.
>
> **Why:** Confidence thresholds are tuning parameters, not implementation details. They belong alongside vote counts and score values so all detection sensitivity can be adjusted in one place.

> **▶ Change Log — per-key thresholds in scoring.py (SCORE_MIN_CONF)**
> `settings/scoring.py` retains a `SCORE_MIN_CONF` dict used by RiskEngine as a scoring gate (distinct from the YOLO detection gate). Previously a single global `MIN_CONF=0.5` was used for all events, meaning low-confidence YOLO detections could score with reduced weight even below the detector's own confidence threshold.

---

## 6. Cooldown System (Revised)

Three cooldown types exist per event key. Two rules are strictly enforced:

```
api_cooldown  == score_cooldown
warn_cooldown <= score_cooldown
```

| Key | score_cooldown | api_cooldown | warn_cooldown |
|---|---|---|---|
| looking_away/down/up/side | 5s | 5s | 3s |
| partial_face | 5s | 5s | 3s |
| face_hidden | 5s | 5s | 3s |
| fake_presence | 10s | 10s | 5s |
| phone | 15s | 15s | 8s |
| multiple_people | 10s | 10s | 5s |
| no_person | 10s | 10s | 5s |
| book | 30s | 30s | 15s |
| headphone | 30s | 30s | 15s |
| earbud | 30s | 30s | 15s |
| speaker_audio | internal tier system (see §4.11) | `SPEAKER_ALERT_COOLDOWN` (10s) | `SPEAKER_WARN_COOLDOWN` (3s) |

> **▶ Change Log:** Previously api_cooldowns were much longer than score_cooldowns (e.g., looking_away: score=3s, api=45s), causing score to accumulate silently for long periods with no on-screen alert. Now every scoring event directly fires an alert.

---

## 7. Score Decay System

Two score buckets exist:

**Non-decaying (fixed) bucket** — permanently accumulated, never reduced:
- phone, fake_presence, multiple_people, no_person, speaker_audio, tab_switch

**Decaying bucket** — reduced periodically:
- gaze events, book, headphone, earbud, face_hidden, partial_face

Decay formula:
```
decay_interval = max(60s, session_duration / 20)
every decay_interval: decaying_score -= 5  (minimum 0)
```

Maximum value per bucket: 150 points.

> **▶ Change Log:** `long speaking` removed from non-decaying list (old speaking detection removed). `speaker_audio` added to non-decaying list. Bucket cap of 150 made explicit in `settings/scoring.py` as `DECAY_BUCKET_CAP`.

---

## 8. Combo Bonuses

Extra score when two suspicious conditions occur simultaneously. Each combo has a 60s internal cooldown.

| Combination | Bonus | Bucket |
|---|---|---|
| looking_down + book | +15 | decaying |
| phone + looking_down | +20 | decaying |

> **▶ Change Log:** `looking_side + speaking` combo removed (old speaking detection removed). Combo values moved to `settings/scoring.py` as `COMBO_DOWN_BOOK` and `COMBO_PHONE_DOWN`.

---

## 9. Risk Thresholds and State Machine

| Score range | State |
|---|---|
| 0–30 | NORMAL |
| 30–60 | WARNING |
| 60–100 | HIGH_RISK |
| ≥ 100 | ADMIN_REVIEW |
| — | TERMINATED |

Termination via score requires Admin Review state. Automatic termination rules bypass score:
- Multiple people detected continuously > 20s → TERMINATE
- No person detected continuously > 20s → TERMINATE

> **▶ Change Log:** Thresholds unchanged. No-person termination rule added (was missing from original design). Both continuous-timer rules include a 1.5s flicker grace period to prevent brief frame dropouts from resetting the timer.

---

## 10. Settings Architecture

> **▶ New Section**

All tunable values are centralized in `settings/` and `config.py`:

```
config.py          — detection toggles, debug gates, YOLO confidence thresholds,
                     vote counts, audio params, proof params, file paths
settings/
  scoring.py       — score values, score cooldowns, state thresholds, decay,
                     combos, speaker audio tier knobs
  alerts.py        — api cooldowns, warn cooldowns, display durations,
                     WARN_MESSAGES + ALERT_MESSAGES dicts
```

**Engine files contain zero hardcoded numbers.** Both import from settings:
- `core/risk_engine.py` → `import settings.scoring as S`
- `core/alert_engine.py` → `import settings.alerts as A`

**Why:** Previously all values were scattered across `risk_engine.py`, `alert_engine.py`, and `config.py`. A change like "increase earbud score from 10 to 20" required editing multiple files and risked missing one. Now one file change affects all dependent logic.

---

## 11. Proof Capture System

> **▶ New Section**

### 11.1 Overview

When `SAVE_PROOF = True` in `config.py`, each scoring alert saves a proof file alongside the report. Proof is categorised by event type into three formats.

> **▶ Change Log — from snapshots to typed proof**
> **Before:** `AlertEngine._save_snapshot()` saved a single JPEG frame for every alert regardless of event type. The path was stored as `"snapshot"` in the alert log entry. AlertEngine mixed file I/O with alert routing logic.
>
> **After:** `ProofWriter` (`utils/proof_writer.py`) handles all proof capture. AlertEngine has zero file I/O. Three proof types are used based on event semantics. Audio is included for speaker events.
>
> **Why:** A single frame is insufficient evidence for time-based events (e.g., `looking_away` — a single frame could be innocent). For speaker audio, having audio alongside video is essential evidence. Separating proof logic from AlertEngine keeps the routing layer clean and testable.

### 11.2 Proof Types

| Type | Format | Events | Details |
|---|---|---|---|
| **image** | `.jpg` | phone, book, headphone, earbud | Single frame at alert time, written synchronously |
| **video** | `.mp4` | looking_away/down/up/side, face_hidden, partial_face, fake_presence | 5s clip centred on alert (2.5s pre + 2.5s post), written async |
| **av** | `.mkv` + `.wav` | speaker_audio | Same 5s window; video + companion audio WAV; merged to single `.mkv` if ffmpeg available |
| **image + video** | `.jpg` per alert + `.mp4` on termination | multiple_people, no_person | Image saved per scoring occurrence; 5s video saved when exam is terminated |

### 11.3 Rolling Frame Buffer

`ProofWriter` maintains a rolling deque of `(timestamp, frame)` pairs (`maxlen=150`, ~5s at 30fps). `push_frame()` is called every main loop iteration.

At alert time, pre-event frames are snapshotted immediately and passed to an async thread. The thread sleeps for `PROOF_POST_S` seconds, then reads post-event frames from the still-rolling buffer, combines them, and writes the file.

### 11.4 Audio Ring Buffer

`AudioMonitor` maintains a timestamped deque of `(wall_time, chunk_bytes)` pairs (~30s of audio). `get_audio_range(t0, t1)` returns all PCM bytes captured in the given wall-clock window.

For AV proof: the async write thread calls `get_audio_range(event_time - pre_s, event_time + post_s)` after sleeping, then writes the WAV and merges (or saves as companion file).

### 11.5 ffmpeg Availability

| ffmpeg present | Output |
|---|---|
| Yes | Single `.mkv` (video + AAC audio merged) |
| No | `.mp4` (video) + `.wav` (audio) as separate companion files |

### 11.6 Termination Deduplication

> **▶ Change Log — duplicate termination alert/proof bug**
> **Before:** Once `RiskEngine.terminated` is set to `True`, every subsequent `process_event()` call in the same frame returns `RiskEvent(terminated=True)` regardless of the key. Both `no_person` and `speaker_audio` (processed in the same frame) generated a termination alert and saved proof. Report contained two "EXAM TERMINATED" entries, one of which had a spurious `speaker_audio_*.mp4` proof path.
>
> **After:** `AlertEngine._termination_alerted` flag (bool) ensures the termination alert is emitted only once. `_termination_proved` flag in `main.py` ensures proof is captured only once, for the key that triggered termination.
>
> **Why:** Multiple termination log entries confuse report readers and waste disk space on spurious proof files.

---

## 12. Session Report

> **▶ Change Log — session folder structure**
> **Before:** Report saved as `reports/report_YYYYMMDD_HHMMSS.json`. Proof snapshots in `reports/snapshots/`. Audio report in `reports/audio/<session_id>/`.
>
> **After:** Each session gets its own folder: `reports/<YYYYMMDD_HHMMSS>/`. Report is `report.json` inside that folder. Proof files are in `reports/<YYYYMMDD_HHMMSS>/proof/`. No separate audio report.
>
> **Why:** A flat `reports/` folder grows unwieldy over multiple sessions. A per-session folder keeps all evidence for one exam together: report + images + videos + audio — easy to archive, share, or delete per candidate.

**Session folder layout:**
```
reports/
  20260312_133718/
    report.json
    proof/
      earbud_133749_59.jpg
      looking_away_133728_08.mp4
      speaker_audio_134012_33.mp4
      speaker_audio_134012_33.wav     ← companion audio (no ffmpeg)
      no_person_133828_58.jpg
      no_person_133838_66.mp4         ← termination video
```

**Report schema:**
```json
{
  "session_start": "2026-03-12 13:37:18",
  "session_end":   "2026-03-12 13:38:42",
  "duration_s":    83.8,
  "total_api_alerts": 9,
  "total_warnings":   13,
  "alert_summary":  { "Candidate not facing screen  [+5 pts]": 3 },
  "warning_summary": { "Not facing screen": 3 },
  "alert_log": [
    {
      "time": "00:09",
      "elapsed_s": 9.6,
      "message": "Candidate not facing screen  [+5 pts]",
      "score_added": 5.0,
      "proof": "reports/20260312_133718/proof/looking_away_133728_08.mp4"
    }
  ],
  "warning_log": [
    { "time": "00:09", "elapsed_s": 9.7, "message": "Not facing screen" }
  ],
  "risk": {
    "final_score": 175.0,
    "fixed_score": 125.0,
    "decaying_score": 50.0,
    "final_state": "TERMINATED",
    "occurrences": { "earbud": 6, "looking_away": 3 },
    "terminated": true,
    "termination_reason": "No person detected >20s",
    "decay_ticks": 0,
    "decay_log": []
  }
}
```

> **▶ Change Log:** `warning_log` added alongside `alert_log` — previously warnings were on-screen only and not recorded. `audio` section removed (old audio proctoring removed). Score preview removed from warnings. `snapshot` field renamed to `proof`. Proof path points into the session folder.

---

## 13. Automatic Termination Rules

Two automatic termination rules exist:

1. **Multiple people detected continuously for > 20 seconds**
2. **No person detected continuously for > 20 seconds**

Both include a 1.5s flicker grace period. Brief detection dropouts (single frames, lighting changes) do not reset the continuous timer.

All other termination passes through the score state machine (ADMIN_REVIEW → TERMINATED).

---

## 14. Temporal Processing Layer

Before risk evaluation, all events pass through temporal filtering:

- **ObjectTemporalTracker:** sliding window vote (default 15 frames, min 5 votes).

  | Object | Vote threshold | Config constant | Reason |
  |---|---|---|---|
  | default | 5/15 | `OBJECT_MIN_VOTES` | — |
  | phone | 9/15 | `PHONE_MIN_VOTES` | brand text / logos cause false positives |
  | book | 10/15 | `BOOK_MIN_VOTES` | paper, notebook surfaces cause false positives |
  | earbud | 9/15 | `EARBUD_MIN_VOTES` | earrings, hair accessories cause false positives |

  > **▶ Change Log:** `BOOK_MIN_VOTES` and `EARBUD_MIN_VOTES` added to `config.py` (previously used default 5/15). Both raised to reduce false-positive detections that were triggering warnings during normal exam conditions.

- **HeadTracker:** duration gate — condition must persist N seconds before triggering (1.5s for head/gaze, 1.0s for looking_side).
- **Flicker frame guard:** frames with mean < 5 or std < 8 skipped (corrupt/black frames).
- **Flicker grace:** multiple_people and no_person continuous timers tolerate 1.5s dropout.

---

## 15. Software Architecture

```
main.py
  ├── ObjectDetector         (YOLO — phone, book, headphone, earbud, person)
  ├── HeadPoseDetector       (MediaPipe — yaw, pitch, gaze, blink, partial_face)
  ├── LipDetector            (MAR + variance + oscillation → is_speaking, is_yawning)
  ├── AudioMonitor           (pyaudio + silero-VAD → speech_active; 30s ring buffer)
  ├── SpeakerAudioDetector   (speech_active + lip_speaking → desync flag)
  ├── ObjectTemporalTracker  (per-key vote window)
  ├── HeadTracker            (duration gate)
  ├── LivenessDetector       (fake presence)
  ├── RiskEngine             (scoring, state, decay, combos)
  ├── AlertEngine            (warn vs alert routing)
  └── ProofWriter            (image / video / AV proof capture)

settings/
  ├── scoring.py
  └── alerts.py
```

**RiskEngine responsibilities:**
- Occurrence counting (rising edge)
- Grace enforcement (occ=1 arms cooldown, no score)
- Two-bucket scoring (fixed + decaying)
- Cooldown tracking
- Duration tracking
- Combo bonuses
- Gaze aggregation bonus
- State machine update
- Termination rules
- Speaker audio tier logic (`_handle_special`)

**AlertEngine responsibilities:**
- `risk_added == 0` → warn
- `risk_added  > 0` → alert (with score added)
- API cooldown gating (= score cooldown)
- Termination alert deduplication (`_termination_alerted` flag)

**ProofWriter responsibilities:**
- Rolling frame buffer (`push_frame`)
- Proof type dispatch (image / video / AV) based on event key
- Async video/AV writing (pre+post frames, audio ring slice)
- ffmpeg merge or companion WAV fallback
- `flush()` on shutdown to complete pending writes

---

## 16. Future Improvements

- Sliding window risk analysis
- Adaptive scoring models based on exam duration
- Admin dashboard with live video and alert stream
- `ffmpeg` installation for single-file AV proof (`.mkv`)
- Behavioral pattern detection (repeated sequences of gaze events)
- Multi-face landmark tracking (distinguish candidate from other people in frame)
