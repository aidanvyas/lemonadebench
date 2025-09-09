from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Interaction(BaseModel):
    attempt: int
    timestamp: str
    request: dict[str, Any]
    response: dict[str, Any]
    tool_executions: list[dict[str, Any]]
    duration_ms: int


class DayRecording(BaseModel):
    day: int
    game_state_before: dict[str, Any]
    interactions: list[Interaction]
    game_state_after: dict[str, Any] | None
    total_attempts: int
    total_duration_ms: int
    start_time: str
    end_time: str | None


class GameRecording(BaseModel):
    game_id: int
    model: str
    start_time: str
    end_time: str | None
    duration_seconds: float | None
    parameters: dict[str, Any]
    days: list[DayRecording]
    final_results: dict[str, Any] | None
    total_tokens: int
    total_cost: float


class BenchmarkMetadata(BaseModel):
    version: str
    timestamp_start: str
    timestamp_end: str | None
    total_duration_seconds: float | None
    parameters: dict[str, Any]


class BenchmarkRecording(BaseModel):
    benchmark_metadata: BenchmarkMetadata
    games: list[GameRecording]
