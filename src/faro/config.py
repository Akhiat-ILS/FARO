"""Explicit, immutable attack settings."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AttackConfig:
    """Linf threat set in raw [0,1] input units.

    Defaults preserve the evaluated ten-block, 100-step CE/DLR schedule.
    There is no global forward-call cutoff. Steps include optimizer iterations,
    not setup or verification forwards. Changing these settings changes the
    evaluated method.
    """

    epsilon: float
    steps_per_block: int = 100
    blocks: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise ValueError("epsilon must be a number in raw [0,1] input units")
        if not math.isfinite(self.epsilon) or not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be finite and between 0 and 1")
        for name in ("steps_per_block", "blocks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1)")
