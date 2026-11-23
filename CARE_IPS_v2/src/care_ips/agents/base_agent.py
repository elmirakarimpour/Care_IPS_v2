from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentVerdict:
    agent_id: str
    verdict: str
    confidence: float
    evidence_ids: frozenset

class DetectionAgent(ABC):
    agent_id: str
    @abstractmethod
    def predict(self, event_features) -> AgentVerdict:
        raise NotImplementedError
