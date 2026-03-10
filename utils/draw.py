import cv2


def draw_alerts(frame, warnings: list[str], alerts: list[str]) -> None:
    """
    warnings — shown in amber/yellow (soft, informational)
    alerts   — shown in red (hard, risk-affecting)
    """
    y = 30
    for msg in warnings:
        cv2.putText(frame, f"[WARN]  {msg}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 200, 255), 1, cv2.LINE_AA)   # amber
        y += 26

    for msg in alerts:
        cv2.putText(frame, f"[ALERT] {msg}",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 255), 2, cv2.LINE_AA)     # red
        y += 26


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} : {det['confidence']:.2f}"

        cv2.rectangle(frame,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=(0, 255, 0),
                      thickness=2)
        cv2.putText(frame,
                    label,
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=1)
