"""Lemonade Stand Economic Reasoning Benchmark."""

# Version 0.5 - Business simulation with inventory management
from .business_game import BusinessGame
from .game_recorder import BenchmarkRecorder, GameRecorder
from .player_factory import PlayerFactory

# Import players if their dependencies are available
try:
    from .openai_player import OpenAIPlayer
except ModuleNotFoundError:  # openai package not installed
    OpenAIPlayer = None

try:
    from .gemini_player import GeminiPlayer
except ModuleNotFoundError:  # google-generativeai package not installed
    GeminiPlayer = None

# New provider players are imported without SDKs at module import time,
# so these imports are safe (their SDKs are lazily imported in __init__).
try:
    from .anthropic_player import AnthropicPlayer
except Exception:  # pragma: no cover - defensive
    AnthropicPlayer = None

try:
    from .xai_player import XAIPlayer
except Exception:  # pragma: no cover - defensive
    XAIPlayer = None

try:
    from .deepseek_player import DeepSeekPlayer
except Exception:  # pragma: no cover - defensive
    DeepSeekPlayer = None

__version__ = "0.5.0"

__all__ = ["BusinessGame", "GameRecorder", "BenchmarkRecorder", "PlayerFactory"]
if OpenAIPlayer is not None:
    __all__.append("OpenAIPlayer")
if GeminiPlayer is not None:
    __all__.append("GeminiPlayer")
if AnthropicPlayer is not None:
    __all__.append("AnthropicPlayer")
if XAIPlayer is not None:
    __all__.append("XAIPlayer")
if DeepSeekPlayer is not None:
    __all__.append("DeepSeekPlayer")
