import time 
import cv2

#handles Time based behavior
class HeadTracker:
    def __init__(self, states, threshold):
        self.states = states
        self.threshold = threshold

    def process(self, frame, key, condition):
        ret_Val = False
        now = time.time()
        this_state = self.states[key]
        label = key.replace("_", " ").title()

        if condition:

            if this_state["start_time"] is None:
                this_state["start_time"] = now

            duration = now - this_state["start_time"]

            if duration >= self.threshold:
                ret_Val = True
        else:
            this_state["start_time"] = None
            this_state["active"] = False

        if this_state["start_time"]:
            elapsed = now - this_state["start_time"]
            cv2.putText(
                    frame,
                    f"{label}: {elapsed:.1f}s",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
            )

        return ret_Val 
