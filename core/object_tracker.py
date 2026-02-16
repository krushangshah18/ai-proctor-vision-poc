from collections import deque

class ObjectTemporalTracker:
    def __init__(self, window=15, min_votes=5):
        self.window = window
        self.min_votes = min_votes
        self.history = {}

    def update(self, key, present):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)

        self.history[key].append(1 if present else 0)

        votes = sum(self.history[key])

        return votes >= self.min_votes
