import experiments.run_benchmark as rb

class DummyPlayer:
    def __init__(self, *args, **kwargs):
        self.closed = False
        DummyPlayer.instance = self
        self.total_token_usage = {"total_tokens": 0}
        self.reasoning_summaries = []
        self.errors = []

    def play_turn(self, game, recorder=None):
        raise RuntimeError("boom")

    def calculate_cost(self):
        return {"total_cost": 0}

    def close(self):
        self.closed = True


def test_close_called_on_exception(monkeypatch):
    monkeypatch.setattr(rb, "OpenAIPlayer", DummyPlayer)
    result = rb.run_single_game(model_name="test-model", game_number=1, days=1)
    assert result["success"] is False
    assert DummyPlayer.instance.closed is True
