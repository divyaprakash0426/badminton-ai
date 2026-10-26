import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import gemini_coach
from engine.analysis import AnalysisEngine
from engine.gemini_coach import GeminiCoach

REPORT = {
    "total_shots": 3,
    "shot_distribution": {"Serve": 1, "Tap Smash": 1, "Lift": 1},
    "rallies": [{
        "winner": "Near", "score_after": "1-0",
        "shots": [
            {"type": "Serve", "full_type": "Short Serve", "hit_by": "Near"},
            {"type": "Lift", "full_type": "Lift", "hit_by": "Far"},
            {"type": "Tap Smash", "full_type": "Tap Smash", "hit_by": "Near"},
        ],
    }],
}


def test_movement_is_per_player_and_in_metres():
    engine = AnalysisEngine.__new__(AnalysisEngine)
    engine.fps = 25.0
    hist = []
    for f in range(50):
        hist.append({"frame": f, "pos": np.array([f * 0.04, 2.0]), "player_type": "Near"})
        hist.append({"frame": f, "pos": np.array([5.0, 12.0 - f * 0.04]), "player_type": "Far"})
    out = AnalysisEngine._analyze_player_movement(engine, hist)
    assert out["player_stats"]["Near"]["total_distance"] == pytest.approx(1.96, abs=0.05)
    assert out["total_distance"] < 5  # interleaving both players would give hundreds of metres
    assert out["max_speed"] == pytest.approx(1.0, abs=0.1)


def test_verdict_normalises_classifier_labels():
    verdict = AnalysisEngine.generate_ai_verdict(REPORT, "Near")
    assert verdict["rally_feedback"][0]["winner"] == "Near"
    assert verdict["score_estimate"] == "1 - 0"


def test_coach_without_key_returns_fallback():
    out = GeminiCoach(None).analyze_match(REPORT)
    assert out["ok"] is False and out["rally_critiques"] == {}


def test_coach_parses_json_and_falls_back_between_models(monkeypatch):
    calls = []

    class FakeModel:
        def __init__(self, name, generation_config=None):
            self.name = name

        def generate_content(self, prompt):
            calls.append(self.name)
            if self.name == "bad-model":
                raise RuntimeError("404 secret-key not found")
            return type("R", (), {"text": '```json\n{"match_summary": "ok", "opponent_weakness": ["a"],'
                                          ' "my_improvements": ["b"], "drills": ["c"], "rally_critiques": {"0": "nice"}}\n```'})()

    monkeypatch.setattr(gemini_coach.genai, "configure", lambda **k: None)
    monkeypatch.setattr(gemini_coach.genai, "GenerativeModel", FakeModel)
    coach = GeminiCoach("secret-key", model="bad-model")
    out = coach.analyze_match(REPORT)
    assert calls[:2] == ["bad-model", gemini_coach.DEFAULT_MODELS[0]]
    assert out["ok"] and out["rally_critiques"] == {"0": "nice"} and out["model"] == gemini_coach.DEFAULT_MODELS[0]


def test_coach_error_never_leaks_key(monkeypatch, capsys):
    class Boom:
        def __init__(self, *a, **k):
            pass

        def generate_content(self, prompt):
            raise RuntimeError("bad request for key secret-key")

    monkeypatch.setattr(gemini_coach.genai, "configure", lambda **k: None)
    monkeypatch.setattr(gemini_coach.genai, "GenerativeModel", Boom)
    out = GeminiCoach("secret-key").analyze_match(REPORT)
    assert out["ok"] is False
    assert "secret-key" not in capsys.readouterr().out
