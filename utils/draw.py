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
        label = f"{det['class']} : {det['confidence']:.2f}"

        cv2.rectangle(frame, 
                      pt1=(x1, y1), 
                      pt2=(x2, y2), 
                      color=(0, 255, 0), 
                      thickness=2)
        cv2.putText(frame, 
                    label, 
                    org=(x1, y1-10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.6, 
                    color=(0, 255, 0), 
                    thickness=1)
