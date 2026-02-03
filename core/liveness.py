import time

class LivenessDetector:
    def __init__(self, window, interval, min_variance, blink_timeout, weights):
        self.window = window
        self.interval = interval
        self.min_variance = min_variance
        self.blink_timeout = blink_timeout
        self.weights = weights

        self.yaw = []
        self.pitch = []
        self.gaze = []

        self.last_blink = time.time()
    
    def _variance(self, values):
        if len(values) < 10:
            return 1.0  # not enough data → assume real
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def update(self, yaw, pitch, gaze, blinked):
        now = time.time()
        
        if not self.yaw or now - self.yaw[-1][0] > self.interval:
            self.yaw.append((now, yaw))
            self.pitch.append((now, pitch))
            self.gaze.append((now, gaze))

        self.yaw = [(t,v) for t,v in self.yaw if now - t <= self.window]
        self.pitch = [(t,v) for t,v in self.pitch if now - t <= self.window]
        self.gaze = [(t,v) for t,v in self.gaze if now - t <= self.window]

        if blinked:
            self.last_blink = now

    def is_fake(self):
        yaw_var = self._variance([v for _, v in self.yaw])
        pitch_var = self._variance([v for _, v in self.pitch])
        gaze_var = self._variance([v for _, v in self.gaze])

        score = (
                self.weights["yaw"] * yaw_var +
                self.weights["gaze"] * gaze_var +
                self.weights["pitch"] * pitch_var
        )
        static = score < self.min_variance

        no_blink = (time.time() - self.last_blink) > self.blink_timeout

        return static and no_blink, (yaw_var, pitch_var, gaze_var)




