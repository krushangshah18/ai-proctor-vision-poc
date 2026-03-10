from collections import deque

class ObjectTemporalTracker:
    def __init__(self, window=15, min_votes=5,
                 per_key_min_votes: dict | None = None):
        self.window    = window
        self.min_votes = min_votes
        # Per-key overrides for min_votes (e.g. stricter threshold for phone)
        self._per_key  = per_key_min_votes or {}
        self.history   = {}

    def update(self, key, present):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)

        self.history[key].append(1 if present else 0)

        votes     = sum(self.history[key])
        threshold = self._per_key.get(key, self.min_votes)
        return votes >= threshold