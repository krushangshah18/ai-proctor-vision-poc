import time
import cv2

# Per-key y-positions so multiple active timers stack vertically without overlap
_TIMER_Y: dict[str, int] = {
    "looking_away" : 230,
    "looking_down"  : 255,
    "looking_up"    : 280,
    "looking_side"  : 305,
    "face_hidden"   : 330,
    "partial_face"  : 355,
    "fake_presence" : 380,
}
_TIMER_Y_DEFAULT_START = 230
_TIMER_Y_STEP          = 25

#handles Time based behavior
class HeadTracker:
    def __init__(self, states, threshold, debug=False):
        self.states = states
        self.threshold = threshold
        self.DEBUG = debug

    def process(self, frame, key, condition, threshold=None):
        ret_Val = False
        now = time.time()
        this_state = self.states[key]
        label = key.replace("_", " ").title()
        active_threshold = threshold if threshold is not None else self.threshold

        if condition:

            if this_state["start_time"] is None:
                this_state["start_time"] = now

            duration = now - this_state["start_time"]

            if duration >= active_threshold:
                ret_Val = True
        else:
            this_state["start_time"] = None
            this_state["active"] = False

        if self.DEBUG and this_state["start_time"]:
            elapsed = now - this_state["start_time"]
            y = _TIMER_Y.get(key, _TIMER_Y_DEFAULT_START)
            cv2.putText(
                    frame,
                    f"{label}: {elapsed:.1f}s",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
            )

        return ret_Val 
