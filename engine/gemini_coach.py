import google.generativeai as genai
import os
import json

class GeminiCoach:
    def __init__(self, api_key):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # User requested Gemini 3.0 / Deep Think
            # We try to use the latest model available in late 2025 context
            try:
                self.model = genai.GenerativeModel('gemini-3-pro-preview')
            except:
                print("Gemini 3.0 not found, falling back to 1.5 Pro")
                self.model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.model = None

    def analyze_match(self, analysis_report, focus_player="Near"):
        """
        Sends aggregated match data to Gemini for high-level coaching advice.
        """
        if not self.model:
            return {
                "opponent_weakness": ["API Key missing. Cannot generate insights."],
                "my_improvements": ["API Key missing. Cannot generate insights."]
            }

        # 1. Prepare Data Prompt
        # Summarize rallies to reduce token count / noise
        match_summary = {
            "focus_player_id": focus_player,
            "total_shots": analysis_report.get('total_shots', 0),
            "stats": analysis_report.get('shot_distribution', {}),
            "rallies": []
        }
        
        # Limit to last 20 rallies to keep context manageable if needed, 
        # but JSON is efficient so usually all is fine.
        raw_rallies = analysis_report.get('rallies', [])
        
        for i, r in enumerate(raw_rallies):
            shots = r.get('shots', [])
            if not shots: continue
            
            # Construct Shot Sequence String for detailed tactical context
            # e.g., "Serve(Near) -> Clear(Far) -> Smash(Near)"
            sequence = []
            for s in shots:
                shot_desc = f"{s['type']}({s['hit_by']})"
                sequence.append(shot_desc)
            
            sequence_str = " -> ".join(sequence)
            
            # Simple heuristic winner (same as analysis.py)
            last_shot = shots[-1]
            last_hitter = last_shot['hit_by']
            shot_type = last_shot['type']
            
            # Simplified winner logic for the Prompt
            winner = "?"
            if shot_type in ['Smash', 'Drive', 'Net', 'Drop']: 
                winner = last_hitter
            elif shot_type in ['Clear', 'Serve', 'Lift']: 
                winner = "Far" if last_hitter == "Near" else "Near" # Opponent wins on out
            
            match_summary['rallies'].append({
                "id": str(i), # Use string ID for JSON compatibility with dict keys
                "winner": winner,
                "sequence": sequence_str,
                "end_type": shot_type
            })

        # 2. Construct Prompt
        prompt = f"""
        You are an expert Badminton Coach. Analyze the following match data for the player identified as '{focus_player}'.
        
        Match Data (JSON):
        {json.dumps(match_summary, indent=2)}
        
        Task:
        1. Identify the Opponent's weaknesses. (Where do they lose points? What shots do they struggle against?)
        2. Suggest specific improvements for '{focus_player}'.
        3. **CRITICAL**: For EACH rally in the list, provide a very short (1 sentence) tactical critique. 
           - If '{focus_player}' WON: Praise the strategy (e.g., "Good smash placement to the backhand").
           - If '{focus_player}' LOST: Explain the mistake (e.g., "Lift was too short, inviting the smash").
           - Reference the 'sequence' to understand the flow.
        
        Output Format:
        Return a valid JSON object with exactly three keys:
        {{
            "opponent_weakness": ["point 1", "point 2"],
            "my_improvements": ["point 1", "point 2"],
            "rally_critiques": {{
                "0": "Analysis for rally id 0...",
                "1": "Analysis for rally id 1...",
                ...
            }}
        }}
        Do not include markdown filtering (```json ... ```). Just the raw JSON string.
        """
        
        # 3. Call API
        try:
            response = self.model.generate_content(prompt)
            # print(f"Gemini Response: {response.text}")
            
            # Sanitization
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text)
            return data
            
        except Exception as e:
            # SECURITY: Sanitize error message to prevent leaking API key
            error_msg = str(e)
            if self.api_key and self.api_key in error_msg:
                error_msg = error_msg.replace(self.api_key, "[REDACTED_API_KEY]")

            print(f"Gemini Error: {error_msg}")

            return {
                "opponent_weakness": ["Error generating insights. Please check console."],
                "my_improvements": ["Error generating insights."],
                "rally_critiques": {}
            }
