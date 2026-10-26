"""Keyword groundedness of Gemini rally critiques: fraction of shot-type terms cited that occur in the rally's detected shots."""
import json, os, re
R = os.path.join(os.path.dirname(__file__), "..", "results")
rep = json.load(open(f"{R}/full_match_report.json"))
out = json.load(open(f"{R}/coach_output.json"))
terms = {"tap smash": "tap smash", "smash": "smash", "drop": "drop", "lift": "lift", "clear": "clear", "block": "block",
         "slice": "slice", "flat": "flat", "drive": "drive", "serve": "serve", "cut": "cut", "rush": "rush"}
cited = grounded = 0
per = []
for k, txt in out["rally_critiques"].items():
    present = " ".join(s["full_type"].lower() for s in rep["rallies"][int(k)]["shots"])
    t = txt.lower().replace("tap smash", "tapsmash")
    hits = [w for w in terms if w != "tap smash" and re.search(rf"\b{w}", t.replace("tapsmash", "smash"))]
    if "tapsmash" in t and "smash" not in hits: hits.append("smash")
    g = [w for w in hits if w in present]
    cited += len(hits); grounded += len(g)
    per.append({"rally": int(k), "cited": hits, "grounded": g})
res = {"rallies": len(per), "terms_cited": cited, "terms_grounded": grounded,
       "precision": grounded / max(cited, 1), "per_rally": per}
json.dump(res, open(f"{R}/llm_groundedness.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_rally"})
