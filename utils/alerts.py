from collections import deque
import time

class AlertManager: 
    def __init__(self, display_duration=2.0):
        self.alerts = deque()
        self.display_duration = display_duration

    def add_alert(self, message):
        self.alerts.append({
            "message": message,
            "timestamp": time.time()
        })
    
    def get_active_alerts(self):
        current_time = time.time()

        while self.alerts and current_time - self.alerts[0]["timestamp"] > self.display_duration:
            self.alerts.popleft()

        return [alert["message"] for alert in self.alerts]
