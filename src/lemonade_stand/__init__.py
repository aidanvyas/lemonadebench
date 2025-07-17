"""Lemonade Stand Economic Reasoning Benchmark."""

# Version 0.5 - Business simulation with inventory management
from .business_game import BusinessGame
from .game_recorder import GameRecorder, BenchmarkRecorder
try:
    from .openai_player import OpenAIPlayer
except ModuleNotFoundError:  # openai package not installed
    OpenAIPlayer = None

__version__ = "0.5.0"

__all__ = ["BusinessGame", "GameRecorder", "BenchmarkRecorder"]
if OpenAIPlayer is not None:
    __all__.insert(1, "OpenAIPlayer")
