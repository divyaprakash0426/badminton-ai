"""Find YouTube sources for ShuttleSet22 matches (the dataset ships without URLs).

For each match in ShuttleSet22's match.csv that is not already covered by ShuttleSet, search
YouTube via yt-dlp (metadata only) and score candidates by player names, tournament/year tokens,
channel and duration. Writes a ranked candidate CSV plus a `urls.txt` of high-confidence picks.

Usage:
    python research/data/find_shuttleset22.py [--out data/shuttleset22] [--min-score 0.75]

High-confidence picks still require Phase 0 verification (court corners vs homography.csv,
hit-frame alignment) before their annotations are trusted.
"""
import argparse
import csv
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SS = os.path.join(ROOT, "CoachAI-Projects", "ShuttleSet", "set", "match.csv")
SS22 = os.path.join(ROOT, "CoachAI-Projects", "CoachAI-Challenge-IJCAI2023", "ShuttleSet22", "set", "match.csv")
STOP = {"bwf", "the", "and", "of", "badminton", "championships", "open", "finals", "final", "hsbc", "world", "tour"}


def tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}


def surnames(name):
    # ShuttleSet writes surnames in upper case (e.g. "NG Ka Long Angus", "Carolina MARIN"); fall back to all tokens.
    upper = [w for w in re.findall(r"[A-Za-z]+", name) if w.isupper() and len(w) > 1]
    return {w.lower() for w in (upper or re.findall(r"[A-Za-z]+", name))}


def search(query, n=8):
    cmd = ["yt-dlp", "--no-update", "--flat-playlist", "-J", f"ytsearch{n}:{query}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        return json.loads(out).get("entries", []) if out else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def score(match, entry):
    title = entry.get("title") or ""
    t = tokens(title)
    player = bool(surnames(match["winner"]) & t) + bool(surnames(match["loser"]) & t)
    tour = tokens(match["tournament"]) - STOP
    tour_hit = len(tour & t) / max(len(tour), 1)
    year_hit = match["tournament"][-4:] in title or match["year"] in title
    channel = "bwf" in (entry.get("channel") or entry.get("uploader") or "").lower()
    dur = entry.get("duration") or 0
    target = float(match["duration"]) * 60
    dur_ok = 0.0 if not dur else max(0.0, 1 - abs(dur - target) / max(target, 1) / 0.5)
    highlight = any(w in title.lower() for w in ("highlight", "shorts", "reaction"))
    return round(0.4 * player / 2 + 0.15 * tour_hit + 0.1 * year_hit + 0.1 * channel + 0.25 * dur_ok
                 - 0.3 * highlight, 3)


def strict_ok(match, r):
    """Hard constraints for an automatic pick; everything else is left for manual review."""
    t = tokens(r["title"] or "")
    tour = tokens(match["tournament"]) - STOP
    ratio = (r["duration"] or 0) / (float(match["duration"]) * 60)
    return (r["channel"] == "BWF TV"
            and bool(surnames(match["winner"]) & t) and bool(surnames(match["loser"]) & t)
            and match["tournament"][-4:] in (r["title"] or "")
            and len(tour & t) / max(len(tour), 1) >= 0.5
            and 1.0 <= ratio <= 2.0)


def find(match):
    q = f"{match['winner']} vs {match['loser']} {match['tournament']} {match['round']}"
    ranked = sorted(({"score": score(match, e), "id": e.get("id"), "title": e.get("title"),
                      "channel": e.get("channel") or e.get("uploader"), "duration": e.get("duration")}
                     for e in search(q) if e.get("id")), key=lambda r: -r["score"])
    return match, q, ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "shuttleset22"))
    ap.add_argument("--min-score", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    covered = {r["video"] for r in csv.DictReader(open(SS, encoding="utf-8"))}
    matches = [r for r in csv.DictReader(open(SS22, encoding="utf-8")) if r["video"] not in covered]
    print(f"{len(matches)} ShuttleSet22 matches not covered by ShuttleSet")

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(find, matches))

    picks, used = [], set()
    with open(os.path.join(args.out, "candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "video", "duration_min", "query", "rank", "score", "youtube_id", "title", "channel",
                    "yt_duration_min", "selected"])
        for match, q, ranked in results:
            best = ranked[0] if ranked else None
            sel = bool(best and best["score"] >= args.min_score and strict_ok(match, best) and best["id"] not in used)
            if sel:
                used.add(best["id"])
                picks.append((match, best))
            for i, r in enumerate(ranked[:3]):
                w.writerow([match["id"], match["video"], match["duration"], q, i, r["score"], r["id"], r["title"],
                            r["channel"], round((r["duration"] or 0) / 60, 1), sel and i == 0])

    with open(os.path.join(args.out, "urls.txt"), "w") as f, \
            open(os.path.join(args.out, "url_map.csv"), "w", newline="", encoding="utf-8") as m:
        w = csv.writer(m)
        w.writerow(["id", "video", "url", "youtube_id", "duration_min", "score", "title"])
        for match, best in picks:
            url = f"https://www.youtube.com/watch?v={best['id']}"
            f.write(url + "\n")
            w.writerow([match["id"], match["video"], url, best["id"], match["duration"], best["score"], best["title"]])
    print(f"selected {len(picks)}/{len(matches)} with score >= {args.min_score}; see {args.out}/candidates.csv")


if __name__ == "__main__":
    main()
