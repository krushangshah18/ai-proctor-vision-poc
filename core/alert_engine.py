import time

class AlertEngine:
    def __init__(self, alert_manager, states, cooldown, reset_cooldown):
        self.alert_manager = alert_manager
        self.states = states
        self.cooldown = cooldown
        self.reset_cooldown = reset_cooldown

    def trigger(self, key, condition):
        now = time.time()
        state = self.states[key]

        if condition:
            if (not state["active"] or (now - state["last_alert"]) > self.cooldown):
                self.alert_manager.add_alert(state["message"])
                state["active"] = True
                state["last_alert"] = now  
        else:
            if state["active"] and (now - state["last_alert"]) > self.reset_cooldown:
                state["active"] = False
