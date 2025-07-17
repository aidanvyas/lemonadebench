from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Result:
    """Standard success result from game operations."""

    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    def asdict(self) -> Dict[str, Any]:
        return {"success": self.success, **self.data}


@dataclass
class GameError(Exception):
    """Exception raised for invalid game operations."""

    message: str
    data: Dict[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def asdict(self) -> Dict[str, Any]:
        result = {"success": False, "error": self.message}
        if self.data:
            result.update(self.data)
        return result
