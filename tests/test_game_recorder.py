import types
import pytest

from src.lemonade_stand.game_recorder import GameRecorder, ValidationError


def _fake_response():
    r = types.SimpleNamespace()
    r.id = "1"
    r.model = "test"
    r.created_at = "now"
    return r


def valid_state():
    return {
        "cash": 100.0,
        "inventory": {},
        "expired_items": {},
        "supply_costs": {},
    }


def valid_interaction():
    return {
        "attempt": 1,
        "request": {"a": 1},
        "response": _fake_response(),
        "tool_executions": [{"tool": "t", "arguments": {}, "result": {}}],
        "duration_ms": 10,
    }


class TestGameRecorder:
    def test_start_day_validation_error(self):
        recorder = GameRecorder(model="m", game_number=1, parameters={})
        with pytest.raises(ValidationError):
            recorder.start_day(1, {"cash": 1})

    def test_record_interaction_validation_error(self):
        recorder = GameRecorder(model="m", game_number=1, parameters={})
        recorder.start_day(1, valid_state())
        bad_exec = [{"tool": "t", "arguments": {}}]  # missing result
        with pytest.raises(ValidationError):
            recorder.record_interaction(
                attempt=1,
                request={},
                response=_fake_response(),
                tool_executions=bad_exec,
                duration_ms=1,
            )

    def test_record_final_results_validation_error(self):
        recorder = GameRecorder(model="m", game_number=1, parameters={})
        recorder.start_day(1, valid_state())
        with pytest.raises(ValidationError):
            recorder.record_final_results({"days_played": 1}, 0.0)

    def test_valid_workflow(self):
        recorder = GameRecorder(model="m", game_number=1, parameters={})
        recorder.start_day(1, valid_state())
        i = valid_interaction()
        recorder.record_interaction(**i)
        recorder.end_day({"cash": 100}, 1)
        results = {
            "days_played": 1,
            "final_cash": 100.0,
            "total_profit": 0.0,
            "total_revenue": 0.0,
            "total_operating_cost": 0.0,
            "total_customers": 0,
            "total_lost_sales": 0,
            "average_daily_profit": 0.0,
            "inventory_value": 0.0,
        }
        recorder.record_final_results(results, 0.0)
        data = recorder.get_recording()
        assert data["final_results"]["final_cash"] == 100.0
