from collections import deque

class RollingErrorHistory:
    """Stores bounded per-agent binary error histories for future CACM updates."""
    def __init__(self, window_size: int = 500):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size
        self._history = {}

    def append(self, agent_id: str, error: int):
        if error not in (0, 1):
            raise ValueError("error must be 0 or 1")
        self._history.setdefault(agent_id, deque(maxlen=self.window_size)).append(error)

    def get(self, agent_id: str):
        return list(self._history.get(agent_id, ()))
