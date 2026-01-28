import cv2

def draw_alerts(frame, alerts):
    y_offset = 30

    for alert in alerts:
        cv2.putText(frame, 
                    alert, 
                    org=(20, y_offset), #x-coordinate, y-coordinate
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.6, #Controls text size : Relative value (not pixels)
                    color=(0,0,255), 
                    thickness=1, 
                    lineType=cv2.LINE_AA
                )
        y_offset += 30

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det["class"]} : {det["confidence"]:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
