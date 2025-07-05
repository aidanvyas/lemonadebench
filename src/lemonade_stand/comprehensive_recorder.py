"""Comprehensive recording of all model interactions for analysis."""

from datetime import datetime
from typing import Any


class ComprehensiveRecorder:
    """Records absolutely everything about model interactions."""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records = []

    def record_request(self, day: int, request_data: dict):
        """Record the full request sent to the API."""
        self.records.append(
            {
                "type": "request",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "data": request_data,
            }
        )

    def record_response(self, day: int, response: Any):
        """Record the full response from the API."""
        # Convert response object to dict
        response_data = {
            "id": response.id if hasattr(response, "id") else None,
            "model": response.model if hasattr(response, "model") else None,
            "created_at": response.created_at
            if hasattr(response, "created_at")
            else None,
            "status": response.status if hasattr(response, "status") else None,
            "output": [],
        }

        # Extract all output items
        if hasattr(response, "output"):
            for item in response.output:
                output_item = {
                    "type": item.type if hasattr(item, "type") else None,
                    "id": item.id if hasattr(item, "id") else None,
                }

                # Handle different output types
                if item.type == "message":
                    output_item["role"] = item.role if hasattr(item, "role") else None
                    output_item["content"] = []
                    if hasattr(item, "content"):
                        for content in item.content:
                            output_item["content"].append(
                                {
                                    "type": content.type
                                    if hasattr(content, "type")
                                    else None,
                                    "text": content.text
                                    if hasattr(content, "text")
                                    else None,
                                }
                            )

                elif item.type == "function_call":
                    output_item["name"] = item.name if hasattr(item, "name") else None
                    output_item["arguments"] = (
                        item.arguments if hasattr(item, "arguments") else None
                    )
                    output_item["call_id"] = (
                        item.call_id if hasattr(item, "call_id") else None
                    )

                elif item.type == "reasoning":
                    output_item["content"] = (
                        item.content if hasattr(item, "content") else None
                    )

                response_data["output"].append(output_item)

        # Extract usage data
        if hasattr(response, "usage"):
            usage = response.usage
            response_data["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "reasoning_tokens": 0,
            }

            if hasattr(usage, "prompt_tokens_details"):
                response_data["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": getattr(
                        usage.prompt_tokens_details, "cached_tokens", 0
                    )
                }

            if hasattr(usage, "output_tokens_details"):
                response_data["usage"]["reasoning_tokens"] = getattr(
                    usage.output_tokens_details, "reasoning_tokens", 0
                )

        # Extract reasoning data
        if hasattr(response, "reasoning"):
            response_data["reasoning"] = {
                "effort": getattr(response.reasoning, "effort", None),
                "summary": getattr(response.reasoning, "summary", None),
            }

        self.records.append(
            {
                "type": "response",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "data": response_data,
            }
        )

    def record_tool_execution(
        self, day: int, tool_name: str, arguments: dict, result: str
    ):
        """Record tool execution details."""
        self.records.append(
            {
                "type": "tool_execution",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    def record_game_state(
        self, day: int, price: float, customers: int, profit: float
    ):
        """Record game state after each turn."""
        self.records.append(
            {
                "type": "game_state",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "price": price,
                "customers": customers,
                "profit": profit,
            }
        )

    def record_error(self, day: int, error: Exception):
        """Record any errors that occur."""
        self.records.append(
            {
                "type": "error",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def get_recording_data(self, test_name: str, model_name: str) -> dict:
        """Get the recording data as a dictionary."""
        return {
            "session_id": self.session_id,
            "test_name": test_name,
            "model_name": model_name,
            "start_time": self.records[0]["timestamp"] if self.records else None,
            "end_time": self.records[-1]["timestamp"] if self.records else None,
            "total_records": len(self.records),
            "records": self.records,
        }
