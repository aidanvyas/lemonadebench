from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Interaction(BaseModel):
    attempt: int
    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    tool_executions: List[Dict[str, Any]]
    duration_ms: int


class DayRecording(BaseModel):
    day: int
    game_state_before: Dict[str, Any]
    interactions: List[Interaction]
    game_state_after: Optional[Dict[str, Any]]
    total_attempts: int
    total_duration_ms: int
    start_time: str
    end_time: Optional[str]


class GameRecording(BaseModel):
    game_id: int
    model: str
    start_time: str
    end_time: Optional[str]
    duration_seconds: Optional[float]
    parameters: Dict[str, Any]
    days: List[DayRecording]
    final_results: Optional[Dict[str, Any]]
    total_tokens: int
    total_cost: float


class BenchmarkMetadata(BaseModel):
    version: str
    timestamp_start: str
    timestamp_end: Optional[str]
    total_duration_seconds: Optional[float]
    parameters: Dict[str, Any]


class BenchmarkRecording(BaseModel):
    benchmark_metadata: BenchmarkMetadata
    games: List[GameRecording]
