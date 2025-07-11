"""Records complete game interactions for reproducibility and analysis."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class GameRecorder:
    """Records all game interactions, API calls, and state changes."""
    
    def __init__(self, model: str, game_number: int, parameters: dict[str, Any]):
        """Initialize recorder for a specific game.
        
        Args:
            model: Model name being tested
            game_number: Game number in the benchmark
            parameters: Benchmark parameters (days, starting_cash, seed, etc.)
        """
        self.model = model
        self.game_number = game_number
        self.parameters = parameters
        
        # Initialize recording structure
        self.start_time = datetime.now()
        self.game_data = {
            "game_id": game_number,
            "model": model,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "parameters": parameters,
            "days": [],
            "final_results": None,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        
        # Track current day being recorded
        self.current_day = None
        self.current_day_data = None
        self.current_attempt = 0
        
    def start_day(self, day_number: int, game_state: dict[str, Any]):
        """Start recording a new day.
        
        Args:
            day_number: Day number (1-based)
            game_state: Complete game state before day starts
        """
        self.current_day = day_number
        self.current_attempt = 0
        self.current_day_data = {
            "day": day_number,
            "game_state_before": game_state,
            "interactions": [],
            "game_state_after": None,
            "total_attempts": 0,
            "total_duration_ms": 0,
            "start_time": datetime.now().isoformat(),
        }
        
    def record_interaction(self, attempt: int, request: dict[str, Any], 
                         response: Any, tool_executions: list[dict[str, Any]],
                         duration_ms: int):
        """Record a complete interaction (request/response/tools).
        
        Args:
            attempt: Attempt number for this day
            request: Complete API request data
            response: API response object
            tool_executions: List of tool calls and results
            duration_ms: Time taken for API call
        """
        if self.current_day_data is None:
            raise ValueError("Must call start_day() before recording interactions")
            
        # Extract response data
        response_data = self._extract_response_data(response)
        
        # Record the interaction
        interaction = {
            "attempt": attempt,
            "timestamp": datetime.now().isoformat(),
            "request": request,
            "response": response_data,
            "tool_executions": tool_executions,
            "duration_ms": duration_ms,
        }
        
        self.current_day_data["interactions"].append(interaction)
        self.current_day_data["total_duration_ms"] += duration_ms
        
        # Update token counts
        if "usage" in response_data:
            self.game_data["total_tokens"] += response_data["usage"].get("total_tokens", 0)
            
    def _extract_response_data(self, response: Any) -> dict[str, Any]:
        """Extract all relevant data from API response object."""
        data = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created_at": getattr(response, "created_at", None),
        }
        
        # Extract reasoning if present
        if hasattr(response, "reasoning") and response.reasoning:
            data["reasoning"] = {
                "content": getattr(response.reasoning, "content", None),
                "effort": getattr(response.reasoning, "effort", None),
            }
            
        # Extract output items
        if hasattr(response, "output"):
            data["output"] = []
            for item in response.output:
                output_item = {
                    "type": getattr(item, "type", None),
                    "id": getattr(item, "id", None),
                }
                
                if item.type == "message":
                    output_item["role"] = getattr(item, "role", None)
                    output_item["content"] = []
                    if hasattr(item, "content"):
                        for content in item.content:
                            output_item["content"].append({
                                "type": getattr(content, "type", None),
                                "text": getattr(content, "text", None),
                            })
                            
                elif item.type == "function_call":
                    output_item["name"] = getattr(item, "name", None)
                    output_item["arguments"] = getattr(item, "arguments", None)
                    output_item["call_id"] = getattr(item, "call_id", None)
                    
                data["output"].append(output_item)
                
        # Extract usage data
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            data["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            }
            
            # Get reasoning tokens if available
            if hasattr(usage, "output_tokens_details"):
                data["usage"]["reasoning_tokens"] = getattr(
                    usage.output_tokens_details, "reasoning_tokens", 0
                )
                
            # Get cached tokens if available
            if hasattr(usage, "prompt_tokens_details"):
                data["usage"]["cached_tokens"] = getattr(
                    usage.prompt_tokens_details, "cached_tokens", 0
                )
                
        return data
        
    def end_day(self, game_state_after: dict[str, Any], total_attempts: int):
        """Finish recording the current day.
        
        Args:
            game_state_after: Complete game state after day ends
            total_attempts: Total attempts made this day
        """
        if self.current_day_data is None:
            raise ValueError("No day in progress")
            
        self.current_day_data["game_state_after"] = game_state_after
        self.current_day_data["total_attempts"] = total_attempts
        self.current_day_data["end_time"] = datetime.now().isoformat()
        
        # Add to days list
        self.game_data["days"].append(self.current_day_data)
        
        # Reset current day tracking
        self.current_day = None
        self.current_day_data = None
        
    def record_final_results(self, results: dict[str, Any], total_cost: float):
        """Record final game results.
        
        Args:
            results: Final game results
            total_cost: Total API cost for this game
        """
        self.game_data["final_results"] = results
        self.game_data["total_cost"] = total_cost
        self.game_data["end_time"] = datetime.now().isoformat()
        self.game_data["duration_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()
        
    def get_recording(self) -> dict[str, Any]:
        """Get the complete recording data."""
        return self.game_data
        
    def save_to_file(self, filepath: Path) -> None:
        """Save recording to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, "w") as f:
            json.dump(self.game_data, f, indent=2)


class BenchmarkRecorder:
    """Records complete benchmark runs with multiple games."""
    
    def __init__(self, parameters: dict[str, Any]):
        """Initialize benchmark recorder.
        
        Args:
            parameters: Benchmark parameters (models, games, days, etc.)
        """
        self.parameters = parameters
        self.start_time = datetime.now()
        
        self.benchmark_data = {
            "benchmark_metadata": {
                "version": "0.5",
                "timestamp_start": self.start_time.isoformat(),
                "timestamp_end": None,
                "total_duration_seconds": None,
                "parameters": parameters,
            },
            "games": [],
        }
        
    def add_game_recording(self, game_recorder: GameRecorder):
        """Add a completed game recording to the benchmark.
        
        Args:
            game_recorder: Completed GameRecorder instance
        """
        self.benchmark_data["games"].append(game_recorder.get_recording())
        
    def finalize(self) -> dict[str, Any]:
        """Finalize the benchmark recording."""
        end_time = datetime.now()
        self.benchmark_data["benchmark_metadata"]["timestamp_end"] = end_time.isoformat()
        self.benchmark_data["benchmark_metadata"]["total_duration_seconds"] = (
            end_time - self.start_time
        ).total_seconds()
        return self.benchmark_data
        
    def save_to_file(self, filepath: Path) -> None:
        """Save complete benchmark recording to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        self.finalize()
        with open(filepath, "w") as f:
            json.dump(self.benchmark_data, f, indent=2)