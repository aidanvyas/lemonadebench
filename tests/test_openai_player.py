import pytest
from openai import OpenAIError

from src.lemonade_stand import openai_player
from src.lemonade_stand.business_game import BusinessGame
from src.lemonade_stand.errors import APICallError, GameError
from src.lemonade_stand.openai_player import OpenAIPlayer


class DummyClient:
    def __init__(self, *_, **__):
        class _R:
            @staticmethod
            def create(**_kw):
                return None

        self.responses = _R()


def test_execute_tool_unknown_tool(monkeypatch):
    monkeypatch.setattr(openai_player, "OpenAI", DummyClient)
    player = OpenAIPlayer(model_name="gpt-4.1-nano", api_key="dummy")
    game = BusinessGame()
    with pytest.raises(GameError):
        player.execute_tool("unknown_tool", {}, game)


def test_play_turn_api_error(monkeypatch):
    monkeypatch.setattr(openai_player, "OpenAI", DummyClient)
    player = OpenAIPlayer(model_name="gpt-4.1-nano", api_key="dummy")
    game = BusinessGame()
    game.start_new_day()

    def fail(**kwargs):
        raise OpenAIError("fail")

    monkeypatch.setattr(player.client.responses, "create", fail)

    with pytest.raises(APICallError):
        player.play_turn(game)
