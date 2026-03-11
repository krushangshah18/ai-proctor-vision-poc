import cv2


def draw_alerts(frame, warnings: list[str], alerts: list[str]) -> None:
    """
    warnings — amber pill badge + amber text on translucent dark background
    alerts   — red pill badge + bright red text on translucent dark background
    """
    import numpy as np
    font     = cv2.FONT_HERSHEY_SIMPLEX
    scale    = 0.48
    thick    = 1
    pad_x    = 10
    pad_y    = 6
    bar_w    = 4    # thin left accent stripe
    gap      = 5
    x0       = 10
    y        = 32
    bg_alpha = 0.72  # background transparency

    def _draw_row(label, label_color, msg, msg_color, bg, accent):
        nonlocal y
        # Measure label and message separately for two-tone text
        (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
        (mw, _),  _ = cv2.getTextSize(msg,   font, scale, thick)
        tw = lw + 8 + mw   # 8px gap between label and message

        row_h  = lh + pad_y * 2
        x1, y1 = x0, y - lh - pad_y
        x2, y2 = x1 + bar_w + pad_x + tw + pad_x, y + pad_y

        # Translucent background
        roi = frame[y1:y2, x1:x2]
        if roi.size:
            overlay = roi.copy()
            overlay[:] = bg
            cv2.addWeighted(overlay, bg_alpha, roi, 1 - bg_alpha, 0, roi)
            frame[y1:y2, x1:x2] = roi

        # Left accent stripe (fully opaque)
        cv2.rectangle(frame, (x1, y1), (x1 + bar_w, y2), accent, -1)

        # Label (e.g. "WARN" / "ALERT") in accent colour
        tx = x1 + bar_w + pad_x
        cv2.putText(frame, label, (tx, y), font, scale, label_color, thick, cv2.LINE_AA)

        # Message in lighter colour
        cv2.putText(frame, msg, (tx + lw + 8, y), font, scale, msg_color, thick, cv2.LINE_AA)

        y += row_h + gap

    # ── Warnings: amber ───────────────────────────────────────────────────────
    for msg in warnings:
        _draw_row(
            label="WARN", label_color=(30, 180, 255),   # amber-orange
            msg=msg,       msg_color=(200, 220, 255),    # soft white-blue
            bg=(10, 25, 45), accent=(30, 150, 255),
        )

    # ── Alerts: red ───────────────────────────────────────────────────────────
    for msg in alerts:
        _draw_row(
            label="ALERT", label_color=(80, 80, 255),   # bright red
            msg=msg,        msg_color=(200, 210, 255),   # soft white
            bg=(10, 10, 40), accent=(60, 60, 240),
        )


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
