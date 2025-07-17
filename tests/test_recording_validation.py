import pytest
from pathlib import Path

from jsonschema import ValidationError

from src.lemonade_stand.game_recorder import GameRecorder, BenchmarkRecorder


def test_game_recorder_invalid(tmp_path: Path) -> None:
    rec = GameRecorder(model="m", game_number=1, parameters={})
    # remove required field
    rec.game_data.pop("game_id")
    with pytest.raises(ValidationError):
        rec.save_to_file(tmp_path / "game.json")


def test_benchmark_recorder_invalid(tmp_path: Path) -> None:
    bench = BenchmarkRecorder(parameters={})
    bench.benchmark_data["games"] = {}
    with pytest.raises(ValidationError):
        bench.save_to_file(tmp_path / "bench.json")
