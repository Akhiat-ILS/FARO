"""Per-image evidence and portable, pickle-free result serialization."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class SampleResult:
    label: int
    clean_prediction: int
    adversarial_prediction: int
    success: bool
    status: str
    linf: float
    margin: float
    blocks_executed: int
    search_forwards: int
    backwards: int
    verification_forwards: int
    seconds: float
    seed: int

    @property
    def total_forwards(self) -> int:
        return self.search_forwards + self.verification_forwards


@dataclass
class AttackResult:
    adversarial: np.ndarray
    samples: tuple[SampleResult, ...]
    config: dict

    @property
    def success(self) -> np.ndarray:
        """True only for clean-correct inputs changed to a wrong prediction."""
        return np.array([s.success for s in self.samples], dtype=bool)

    @property
    def success_rate(self) -> float:
        """Fraction of initially correct inputs fooled; NaN if none eligible."""
        eligible = sum(s.clean_prediction == s.label for s in self.samples)
        return float(self.success.sum() / eligible) if eligible else float("nan")

    def to_dict(self) -> dict:
        eligible = sum(s.clean_prediction == s.label for s in self.samples)
        return {
            "method": "FARO fixed CE/DLR v0.1.0",
            "config": self.config,
            "samples": [asdict(s) | {"total_forwards": s.total_forwards} for s in self.samples],
            "summary": {
                "inputs": len(self.samples), "clean_correct": eligible,
                "successes": int(self.success.sum()),
                "success_rate_on_clean_correct": self.success_rate if eligible else None,
                "search_forwards": sum(s.search_forwards for s in self.samples),
                "verification_forwards": sum(s.verification_forwards for s in self.samples),
                "total_forwards": sum(s.total_forwards for s in self.samples),
                "backwards": sum(s.backwards for s in self.samples),
            },
        }

    def save(self, directory: str | Path) -> None:
        """Create a new directory containing arrays.npz and results.json.

        Existing paths are rejected to avoid overwriting previous experiments.
        Arrays never use pickle. The original images are not copied into results.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "arrays.npz", adversarial=self.adversarial,
                            labels=np.array([s.label for s in self.samples]), success=self.success)
        (directory / "results.json").write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
