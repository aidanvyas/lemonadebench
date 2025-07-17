import json
import sys
from pathlib import Path

import pytest

from experiments import run_benchmark
from src.lemonade_stand.game_recorder import GameRecorder


def stub_run_single_game(model_name: str, game_number: int, days: int = 30, starting_cash: float = 1000, seed: int | None = None):
    rec = GameRecorder(model_name, game_number, {"days": days, "starting_cash": starting_cash, "seed": seed})
    rec.record_final_results(
        {
            "days_played": days,
            "final_cash": starting_cash,
            "total_profit": 0,
            "total_revenue": 0,
            "total_operating_cost": 0,
            "total_customers": 0,
            "average_daily_profit": 0,
            "inventory_value": 0,
        },
        0.0,
    )
    return {
        "game_number": game_number,
        "model": model_name,
        "success": True,
        "starting_cash": starting_cash,
        "days_played": days,
        "final_cash": starting_cash,
        "total_profit": 0,
        "total_revenue": 0,
        "total_operating_cost": 0,
        "total_customers": 0,
        "total_customers_lost": 0,
        "days_with_stockouts": 0,
        "stockout_rate": 0.0,
        "average_daily_profit": 0,
        "final_inventory_value": 0,
        "total_expired_items": {"cups": 0, "lemons": 0, "sugar": 0, "water": 0},
        "total_expired_value": 0,
        "daily_cash_history": [],
        "average_turn_attempts": 0,
        "token_usage": {"total_tokens": 0},
        "cost_info": {"total_cost": 0.0},
        "reasoning_summaries": [],
        "ai_errors": [],
        "duration_seconds": 0.1,
        "game_history": [],
        "supply_cost_history": [],
        "recorder": rec,
    }


def test_parallel_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmark, "run_single_game", stub_run_single_game)
    monkeypatch.chdir(tmp_path)

    args = [
        "run_benchmark.py",
        "--games",
        "2",
        "--days",
        "1",
        "--models",
        "A",
        "B",
        "--starting-cash",
        "100",
        "--no-analysis",
    ]
    monkeypatch.setattr(sys, "argv", args)

    run_benchmark.main()

    files = list((tmp_path / "results" / "json").glob("*_full.json"))
    assert len(files) == 1
    data = json.loads(Path(files[0]).read_text())
    assert len(data["games"]) == 4
