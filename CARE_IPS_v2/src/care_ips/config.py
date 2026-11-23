from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CIC-IDS2017"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "CIC-IDS2017"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim" / "CIC-IDS2017"
RESULTS_DIR = PROJECT_ROOT / "results"

@dataclass(frozen=True)
class CACMConfig:
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25
    w4: float = 0.25
    def __post_init__(self):
        assert abs(self.w1 + self.w2 + self.w3 + self.w4 - 1.0) < 1e-6

@dataclass(frozen=True)
class SafetyGateConfig:
    ir_threshold: float = 0.5
    uncertainty_bound: float = 0.3

@dataclass(frozen=True)
class RollbackConfig:
    window_size: int = 500

@dataclass(frozen=True)
class SamplingConfig:
    sample_size_per_file: int = 20_000
    random_seed: int = 42

CACM = CACMConfig()
SAFETY_GATE = SafetyGateConfig()
ROLLBACK = RollbackConfig()
SAMPLING = SamplingConfig()
LABEL_COL = "Label"
GLOBAL_TEST_PATH = RAW_DATA_DIR / "Global test set.csv"
