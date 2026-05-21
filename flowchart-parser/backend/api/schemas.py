"""Pydantic schemas for the parse API."""

from pydantic import BaseModel, Field


class ParseConfig(BaseModel):
    min_ocr_confidence: float = Field(0.25, ge=0.0, le=1.0)
    ocr_padding: int = Field(3, ge=0, le=30)
    los_threshold: float = Field(0.12, ge=0.01, le=0.5)
    merge_threshold: float = Field(1.0, ge=0.5, le=2.0)
    barrier_sensitivity: float = Field(1.0, ge=0.5, le=2.0)


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "flowchart-text-engine"
