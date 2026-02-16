import cv2

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
