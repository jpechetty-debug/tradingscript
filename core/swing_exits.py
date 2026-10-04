"""Prespecified research-only exit policies; application exits stay unchanged."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExitPolicy:
    mode: str = "FIXED"
    atr_multiple: float = 2.

    def __post_init__(self) -> None:
        if self.mode not in {"FIXED", "ATR_TRAIL", "CHANDELIER", "SCALE_OUT"}:
            raise ValueError("Unknown research exit policy")
        if not 0 < self.atr_multiple < 20:
            raise ValueError("Invalid trailing ATR multiple")
