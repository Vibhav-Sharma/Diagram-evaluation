"""Runtime tuning parameters (API / frontend sliders)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PipelineConfig:
    """Tunable pipeline thresholds."""

    min_ocr_confidence: float = 0.25
    ocr_padding: int = 3
    los_threshold: float = 0.12
    merge_threshold: float = 1.0
    barrier_sensitivity: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PipelineConfig":
        if not data:
            return cls()
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})
