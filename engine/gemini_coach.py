import json
import os
import re

import google.generativeai as genai

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]


def _redact(text, key):
    return text.replace(key, "[REDACTED_API_KEY]") if key and key in text else text


def _strip_fences(text):
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


class GeminiCoach:
    """Tactical coaching on top of the analysis report, backed by Gemini.

    Model order: constructor arg, then $GEMINI_MODEL, then DEFAULT_MODELS.
    Each model is tried in turn so a retired/unavailable model does not break the app.
    """

    def __init__(self, api_key, model=None):
        self.api_key = api_key
        self.last_error = None
        self.active_model = None
        self.models = []
        if not api_key:
            return
        genai.configure(api_key=api_key)
        preferred = model or os.environ.get("GEMINI_MODEL")
        self.models = [preferred] if preferred else []
        self.models += [m for m in DEFAULT_MODELS if m not in self.models]

    @property
    def available(self):
        return bool(self.models)

    @staticmethod
    def build_match_summary(report, focus_player="Near"):
        rallies = []
        for i, r in enumerate(report.get("rallies", [])):
            shots = r.get("shots", [])
            if not shots:
                continue
            rallies.append({
                "id": str(i),
                "shot_count": len(shots),
                "winner": r.get("winner", "?"),
                "score_after": r.get("score_after"),
                "sequence": " -> ".join(
                    f"{s.get('full_type') or s.get('type')}({s.get('hit_by')})" for s in shots
                ),
            })
        return {
            "focus_player": focus_player,
            "total_shots": report.get("total_shots", 0),
            "shot_distribution": dict(report.get("shot_distribution", {})),
            "movement": report.get("player_stats", {}),
            "rallies": rallies,
        }

    def _generate(self, prompt, as_json):
        config = {"temperature": 0.4}
        if as_json:
            config["response_mime_type"] = "application/json"
        errors = []
        for name in self.models:
            try:
                model = genai.GenerativeModel(name, generation_config=config)
                response = model.generate_content(prompt)
                self.active_model = name
                return response.text
            except Exception as e:  # try the next model
                errors.append(f"{name}: {_redact(str(e), self.api_key)[:200]}")
        raise RuntimeError("; ".join(errors))

    def analyze_match(self, analysis_report, focus_player="Near"):
        if not self.available:
            return self._fallback("API key missing. Add a Gemini key to unlock AI coaching.")

        summary = self.build_match_summary(analysis_report, focus_player)
        if not summary["rallies"]:
            return self._fallback("No rallies were detected, so there is nothing to coach yet.")

        prompt = f"""You are an expert badminton coach. Analyse this singles match for the player
identified as '{focus_player}' ('Near' = camera-side player, 'Far' = opposite side).
Shot labels come from a video classifier and can be noisy; do not over-claim certainty.

Match data (JSON):
{json.dumps(summary, indent=2)}

Return a JSON object with exactly these keys:
- "match_summary": 2-3 sentence overview of how the match is going for {focus_player}.
- "opponent_weakness": list of 2-4 short, specific weaknesses of the opponent.
- "my_improvements": list of 2-4 short, specific improvements for {focus_player}.
- "drills": list of 2-3 practice drills tied to the improvements.
- "rally_critiques": object mapping each rally "id" to ONE sentence of tactical critique
  that references the shot sequence. Praise if {focus_player} won, explain the mistake if lost.
"""
        try:
            data = json.loads(_strip_fences(self._generate(prompt, as_json=True)))
            if not isinstance(data, dict):
                raise ValueError("model did not return a JSON object")
        except Exception as e:
            self.last_error = _redact(str(e), self.api_key)
            print(f"Gemini Error: {self.last_error}")
            return self._fallback("AI coach is unavailable right now. Showing rule-based tips instead.")

        self.last_error = None
        return {
            "ok": True,
            "model": self.active_model,
            "match_summary": str(data.get("match_summary", "")),
            "opponent_weakness": [str(x) for x in data.get("opponent_weakness", [])],
            "my_improvements": [str(x) for x in data.get("my_improvements", [])],
            "drills": [str(x) for x in data.get("drills", [])],
            "rally_critiques": {str(k): str(v) for k, v in (data.get("rally_critiques") or {}).items()},
        }

    def ask(self, question, analysis_report, focus_player="Near", history=None):
        """Answer a free-form follow-up question grounded in the match data."""
        if not self.available:
            return "Add a Gemini API key to chat with the coach."
        summary = self.build_match_summary(analysis_report, focus_player)
        turns = "\n".join(f"{role}: {text}" for role, text in (history or [])[-6:])
        prompt = f"""You are an expert badminton coach answering a follow-up question about one match.
Use only the data below; if it cannot answer the question, say so. Be concise (max 120 words).

Match data (JSON):
{json.dumps(summary)}

Conversation so far:
{turns}

Player: {question}
Coach:"""
        try:
            return self._generate(prompt, as_json=False).strip()
        except Exception as e:
            self.last_error = _redact(str(e), self.api_key)
            print(f"Gemini Error: {self.last_error}")
            return "Sorry, the coach could not answer right now. Please try again."

    @staticmethod
    def _fallback(message):
        return {
            "ok": False,
            "model": None,
            "match_summary": "",
            "opponent_weakness": [message],
            "my_improvements": [message],
            "drills": [],
            "rally_critiques": {},
        }
