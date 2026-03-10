# AI Proctoring Risk Scoring System Design

## 1. Purpose of This Document

This document defines the complete scoring and risk evaluation system used by the AI proctoring platform. The purpose of the scoring system is to:

1. Detect cheating behavior reliably.
2. Avoid false positives that may penalize honest students.
3. Provide transparent reasoning for every decision.
4. Allow administrators to review suspicious activity.
5. Terminate exams only when strong evidence of cheating exists.

The system is intentionally designed as a **risk accumulation engine rather than a single-event trigger system**. This ensures that no candidate is penalized due to momentary detection errors.

---

# 2. Core Design Philosophy

During system design several key principles were established.

### 2.1 No Single Frame Decisions

Computer vision models occasionally produce noisy predictions. Therefore:

- No decision is made from a single frame.
- All events must persist for a defined duration before triggering.

This prevents false positives caused by:

- lighting changes
- momentary head movement
- detection glitches

---

### 2.2 Warnings Before Penalties

Many behaviors are not cheating but may indicate poor exam posture.

Examples:

- sitting too far from camera
- adjusting position
- briefly looking away

Therefore the system first provides **soft warnings** before risk scores are applied.

---

### 2.3 Progressive Escalation

Instead of large immediate penalties the system escalates gradually.

Example escalation pattern:

1. warning
2. minor penalty
3. stronger penalty
4. administrator review
5. termination

This approach ensures fairness.

---

### 2.4 Duration Based Decisions

Human behavior is continuous. Therefore many events are evaluated using **duration thresholds**.

Example:

- speaking for 0.5 seconds may be noise
- speaking for 10 seconds is suspicious

Thus duration thresholds are applied to multiple alerts.

---

### 2.5 Risk Score Instead of Immediate Termination

All suspicious activities accumulate into a **risk score**.

Termination is only considered once:

Risk Score >= 100

This ensures multiple suspicious behaviors are required before action.

---

# 3. Alert Tier Classification

Alerts were categorized into three severity tiers.

This classification was determined after discussion about detection reliability and cheating likelihood.

## Tier 3 – Critical Events

These events strongly indicate cheating or exam policy violation.

Events:

- Tab switching
- Multiple people detected
- No person detected
- Mobile phone detected

These signals are considered high severity due to their direct relationship with cheating.

---

## Tier 2 – Suspicious Events

These behaviors may indicate cheating but can also occur naturally.

Events:

- Book detected
- Fake presence (static image)
- Speaking detected
- Face hidden
- Headphones / earbuds

These events accumulate risk gradually.

---

## Tier 1 – Behavioral Signals

These signals are weak indicators individually.

Events:

- Looking away
- Looking down
- Looking up
- Looking side
- Partial face

These signals only contribute small amounts of risk and are mainly useful when repeated.

---

# 4. Detailed Event Policies

## 4.1 Tab Switching

Tab switching is the strongest browser-based cheating indicator.

### Policy

1st occurrence → warning

2nd occurrence → warning

3rd occurrence within 10 minutes → terminate exam

OR

5 tab switches across entire exam → terminate

### Reasoning

Students may accidentally switch tabs once or twice. However repeated switching strongly indicates accessing external resources.

---

## 4.2 Exit Fullscreen

Exiting fullscreen is required to switch tabs.

Therefore it is treated as a **minor indicator rather than a strong violation**.

### Policy

1st occurrence → warning

Subsequent occurrences → +5 risk

Trigger only if fullscreen exit lasts more than 2 seconds.

### Reasoning

Students sometimes accidentally exit fullscreen. Penalizing heavily would be unfair.

---

## 4.3 Multiple People Detection

Detects presence of additional individuals in the camera frame.

### Policy

1st detection → warning

2nd detection → +10 risk

3rd detection → +50 risk

If multiple people detected continuously for more than 20 seconds → terminate exam.

### Reasoning

Short misdetections may occur due to shadows or background artifacts. Sustained presence strongly indicates assistance.

---

## 4.4 No Person Detected

Triggered when no face is visible.

### Policy

5–10 seconds → +25 risk

>10 seconds → +50 risk

>20 seconds → terminate

### Reasoning

Candidates may briefly leave frame while adjusting seating.

Long absence indicates potential consultation with external resources.

---

## 4.5 Mobile Phone Detection

Mobile phones are a strong cheating tool.

### Policy

1st detection → warning

2nd detection → +25 risk

3rd detection → +50 risk

### Additional rule

Phone + looking_down combination → additional +20 risk

### Reasoning

Phone usage during exams is highly correlated with cheating.

---

## 4.6 Fake Presence (Static Image)

Detects scenarios where a candidate replaces themselves with a photo or static frame.

### Policy

<15 seconds static → warning

>15 seconds static → +30 risk

>40 seconds static → +60 risk

### Additional mechanism

System requests candidate to move slightly after warning.

If motion detected → reset timer.

### Reasoning

Candidates may remain still temporarily.

True static images show no micro movements such as:

- blinking
- head pose changes
- facial landmark motion

---

## 4.7 Speaking Detection

Audio analysis detects speaking.

### Policy

>1.5 seconds → warning

>3 seconds → +10 risk

>5 seconds → +20 risk

>10 seconds → +30 risk

### Additional rule

If different voice detected → +40 risk

### Reasoning

Short utterances can occur naturally.

Long speech indicates potential conversation.

---

## 4.8 Face Hidden

Triggered when the face becomes obstructed.

### Policy

>2 seconds → warning

>5 seconds → +10 risk

>10 seconds → +20 risk

### Reasoning

Students may cover their face momentarily while thinking.

Long concealment suggests suspicious activity.

---

## 4.9 Headphones / Earbuds

Audio devices may allow communication with others.

### Policy

1st detection → warning

Subsequent detection → +10 risk

Cooldown applied: 120 seconds

### Reasoning

Headphones may sometimes be misdetected by vision models.

Cooldown prevents repeated penalties.

---

## 4.10 Gaze Direction Signals

Includes:

- looking away
- looking down
- looking side
- looking up

### Policy

Each event → +5 risk

Aggregation rule:

5 gaze events within 30 seconds → additional +10 risk

### Reasoning

Frequent gaze shifts may indicate consulting materials.

---

## 4.11 Partial Face

Triggered when the face is partially outside frame.

### Policy

>3 seconds → +2 risk

### Reasoning

Usually caused by improper camera placement.

---

# 5. Confidence Weighting

Events only contribute risk if detection confidence exceeds threshold.

Example threshold:

confidence >= 0.5

Risk calculation:

Risk += BaseScore × Confidence

Example:

Phone detected

BaseScore = 50
Confidence = 0.8

Risk added = 40

---

# 6. Cooldown System

Each event type has cooldown to prevent repeated scoring.

Example cooldown table:

Gaze events → 3 seconds

Speaking → 5 seconds

Face hidden → 10 seconds

Book detection → 60 seconds

Headphones → 120 seconds

---

# 7. Score Decay System

Not all events decay.

## Non-decaying events

- tab switch
- mobile phone
- fake presence
- multiple people
- long speaking

## Decaying events

- gaze
- partial face
- headphone
- book
- face hidden

Decay interval determined relative to exam duration.

Example:

Decay interval = exam_duration / 20

Each decay:

RiskScore -= 5

Minimum risk = 0

---

# 8. Event Combination Detection

Certain combinations increase suspicion.

Examples:

Looking_down + book → bonus risk

Looking_side + speaking → bonus risk

Phone + looking_down → bonus risk

Typical bonus risk = +15

---

# 9. Risk Thresholds and State Machine

Risk levels determine system state.

0–30 → Normal

30–60 → Warning

60–100 → High Risk

>=100 → Admin Review

Admin confirmed → Termination

---

# 10. Administrator Review System

When risk score reaches threshold:

Admin is notified.

Admin review timeout determined by exam length.

Example:

1 hour exam → 5 minute response window

If another alert occurs during this period → automatic termination.

---

# 11. Automatic Termination Rules

Only two automatic rules exist.

Multiple people detected continuously for >20 seconds

Tab switching threshold reached

All other decisions pass through risk scoring system.

---

# 12. Temporal Processing Layer

Before risk evaluation events pass through temporal filtering.

This layer handles:

- duration checks
- cooldowns
- frame consistency

This ensures stability of detection signals.

---

# 13. Post Exam Risk Report

After exam completion a report is generated.

Report includes:

- event timeline
- risk score progression
- snapshots
- audio segments

Example timeline:

00:05 phone detected

01:23 speaking detected

02:10 looking_down

05:30 multiple_people

This enables transparent review.

---

# 14. Software Architecture Requirements

The scoring system requires supporting infrastructure.

## Event Stream

Detectors must emit structured events.

Example event format:

{
  type: "phone_detected",
  confidence: 0.82,
  duration: 2.4,
  timestamp: 123.2
}

---

## Risk Engine

Central component responsible for score updates.

Responsibilities:

- score calculation
- decay
- cooldown tracking
- escalation logic

---

## State Manager

Maintains current exam state.

Possible states:

NORMAL

WARNING

HIGH_RISK

ADMIN_REVIEW

TERMINATED

---

## Admin Monitoring System

Dashboard must display:

- live video
- alert stream
- risk score
- timeline

---

# 15. Future Improvements

Planned improvements include:

Sliding window risk analysis

Advanced behavioral modeling

Adaptive scoring models

Improved fake presence detection

---

# 16. Conclusion

This scoring system balances fairness and security.

The design ensures:

- minimal false positives
- strong cheating detection
- administrator oversight
- transparent reporting

By combining computer vision, audio analysis, browser monitoring, and a robust risk engine the system provides reliable automated proctoring while preserving candidate fairness.

