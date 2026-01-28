import cv2


def main():
    print("Hello from ai-proctor-vision-poc!")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could Not open WebCam")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("AI Proctoring Vision POC", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
