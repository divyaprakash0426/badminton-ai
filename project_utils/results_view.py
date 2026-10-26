"""Results dashboard (Overview / Rallies / Video / AI Coach) rendered from analysis results."""
import json
import os
import subprocess
import tempfile
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st

from engine.analysis import AnalysisEngine
from engine.gemini_coach import GeminiCoach
from project_utils import ui_theme as ui

_COLOR = alt.Scale(domain=["Near", "Far"], range=[ui.NEAR, ui.FAR])


def _fmt_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _shot_table(report):
    rows = []
    for rally in report.get("rallies", []):
        for s in rally.get("shots", []):
            rows.append({"Shot": s.get("type", "Unknown"), "Player": s.get("hit_by", "?")})
    return pd.DataFrame(rows)


def _overview(report, fps):
    rallies = report.get("rallies", [])
    stats = report.get("player_stats", {})
    near_won = sum(1 for r in rallies if r.get("winner") == "Near")
    far_won = sum(1 for r in rallies if r.get("winner") == "Far")
    avg_len = (sum(r.get("shot_count", 0) for r in rallies) / len(rallies)) if rallies else 0
    ui.kpi_row([
        ("Rallies", len(rallies), f"avg {avg_len:.1f} shots each"),
        ("Total shots", report.get("total_shots", 0), "classified by SlowFast"),
        ("Score estimate", f"{near_won} – {far_won}", "Near – Far (heuristic)"),
        ("Distance", f"{report.get('total_distance', 0):.0f} m", "both players combined"),
        ("Top speed", f"{report.get('max_speed', 0):.1f} m/s", "smoothed footwork peak"),
    ])

    df = _shot_table(report)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Shot mix by player")
        if df.empty:
            ui.empty_state("No shots detected", "Try a different calibration frame or a clearer camera angle.")
        else:
            counts = df.groupby(["Shot", "Player"]).size().reset_index(name="Count")
            order = df["Shot"].value_counts().index.tolist()
            chart = (
                alt.Chart(counts)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    y=alt.Y("Shot:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                    x=alt.X("Count:Q", title="Shots", axis=alt.Axis(tickMinStep=1, format="d")),
                    color=alt.Color("Player:N", scale=_COLOR, legend=alt.Legend(orient="bottom")),
                    tooltip=["Shot", "Player", "Count"],
                )
                .properties(height=max(180, 34 * len(order)))
            )
            st.altair_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Footwork: Near vs Far")
        if stats:
            mv = pd.DataFrame(
                [{"Player": p, "Distance (m)": v["total_distance"], "Top speed (m/s)": v["max_speed"],
                  "Avg speed (m/s)": v["avg_speed"]} for p, v in stats.items()]
            )
            st.dataframe(
                mv.round(2), hide_index=True, use_container_width=True,
                column_config={"Player": st.column_config.TextColumn()},
            )
            dist = alt.Chart(mv).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Player:N", title=None), y=alt.Y("Distance (m):Q"),
                color=alt.Color("Player:N", scale=_COLOR, legend=None),
            ).properties(height=160)
            st.altair_chart(dist, use_container_width=True)
        else:
            st.caption("No movement data available.")

    st.download_button(
        "⬇️ Download report (JSON)",
        data=json.dumps(report, default=str, indent=2),
        file_name="match_report.json",
        mime="application/json",
    )


def _rallies(results, report, info, llm_critiques, verdict):
    rallies = report.get("rallies", [])
    if not rallies:
        ui.empty_state("No rallies found", "The shuttle was not tracked long enough to form a rally.")
        return
    feedback = {f["rally_index"]: f for f in verdict["rally_feedback"]}
    clips_dir = results.get("clips_dir")
    if not clips_dir or not os.path.exists(clips_dir):
        clips_dir = tempfile.mkdtemp(prefix="rally_clips_")
        results["clips_dir"] = clips_dir
    fps = info["fps"]
    near_won = {i for i, r in enumerate(rallies) if r.get("winner") == "Near"}
    st.caption(f"{len(rallies)} rallies · {len(near_won)} won by Near · expand a rally for its clip and shot list")

    for i, rally in enumerate(rallies):
        start = rally.get("start_frame", 0) / fps
        duration = (rally.get("end_frame", 0) - rally.get("start_frame", 0)) / fps
        winner = rally.get("winner", "?")
        title = (f"Rally {i + 1} · {rally.get('shot_count', 0)} shots · {_fmt_time(start)} "
                 f"({duration:.1f}s) · point: {winner} · {rally.get('score_after', '')}")
        with st.expander(title):
            critique = llm_critiques.get(str(i))
            if critique:
                ui.card(f"🤖 {critique}", "near")
            elif feedback.get(i):
                tone = "near" if feedback[i]["winner"] == "Near" else "warn"
                ui.card(feedback[i]["comment"].replace("**", ""), tone)

            clip_path = os.path.join(clips_dir, f"rally_{i + 1}.mp4")
            if not os.path.exists(clip_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", results["converted_file"],
                     "-t", f"{duration:.2f}", "-c", "copy", clip_path],
                    stderr=subprocess.DEVNULL,
                )
            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                st.video(clip_path)
            else:
                st.warning("Clip extraction failed for this rally.")

            shots = rally.get("shots", [])
            if shots:
                st.dataframe(
                    [{
                        "#": n + 1,
                        "Player": ("🟢 " if s.get("hit_by") == "Near" else "🔵 ") + str(s.get("hit_by", "?")),
                        "Shot": s.get("full_type", s.get("type")),
                        "Time": _fmt_time(int(s["from_frame"]) / fps),
                        "From → To": f"{s.get('from_zone', '–')} → {s.get('to_zone', '–')}",
                    } for n, s in enumerate(shots)],
                    use_container_width=True, hide_index=True,
                )


def _video(results):
    path = results["converted_file"]
    st.video(path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button("⬇️ Download analyzed video", f, "badminton_analysis_video.mp4", "video/mp4")


def _coach(report, api_key):
    if not api_key:
        ui.empty_state(
            "Unlock AI coaching",
            "Add a Gemini API key in the sidebar (or set GEMINI_API_KEY) to get a match summary, "
            "opponent weaknesses, drills, per-rally critiques and a chat with your coach.",
        )
        return

    focus_label = st.radio("Coach me as", ["Near player", "Far player"], horizontal=True, key="coach_focus_label")
    focus = "Near" if focus_label.startswith("Near") else "Far"
    insights = st.session_state.get("coach_insights")
    stale = insights is not None and st.session_state.get("coach_focus") != focus

    label = "Regenerate coach analysis" if insights else "Generate coach analysis"
    if st.button(label, type="primary"):
        with st.spinner("Consulting the AI coach…"):
            insights = GeminiCoach(api_key).analyze_match(report, focus_player=focus)
        st.session_state.coach_insights = insights
        st.session_state.coach_focus = focus
        st.session_state.coach_chat = []
        stale = False
    if not insights:
        st.caption("Press the button to analyze this match with Gemini.")
        return
    if stale:
        st.warning(f"These insights were written for the {st.session_state.get('coach_focus')} player. Regenerate to switch.")

    if not insights.get("ok"):
        ui.card(insights["opponent_weakness"][0], "warn")
    else:
        st.caption(f"Model: {insights.get('model')}")
        if insights.get("match_summary"):
            ui.card(insights["match_summary"], "near")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎯 Opponent weaknesses")
        for item in insights["opponent_weakness"]:
            ui.card(item, "far")
    with c2:
        st.markdown("#### 📈 Your improvements")
        for item in insights["my_improvements"]:
            ui.card(item, "near")
    if insights.get("drills"):
        st.markdown("#### 🏋️ Suggested drills")
        for item in insights["drills"]:
            ui.card(item)

    if insights.get("ok"):
        st.markdown("#### 💬 Ask the coach")
        chat = st.session_state.setdefault("coach_chat", [])
        for role, text in chat:
            with st.chat_message("user" if role == "Player" else "assistant"):
                st.write(text)
        question = st.chat_input("e.g. How do I defend their smash better?")
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = GeminiCoach(api_key).ask(question, report, st.session_state.get("coach_focus", focus), chat)
                st.write(answer)
            chat += [("Player", question), ("Coach", answer)]


def render_results(results, info, api_key):
    report = results["analysis_report"]
    insights = st.session_state.get("coach_insights") or {}
    verdict = AnalysisEngine.generate_ai_verdict(report, focus_player=st.session_state.get("coach_focus", "Near"))

    tab_over, tab_rallies, tab_video, tab_coach = st.tabs(
        ["📊 Overview", "🏸 Rallies", "🎥 Video", "🤖 AI Coach"]
    )
    with tab_over:
        _overview(report, info["fps"])
    with tab_rallies:
        _rallies(results, report, info, insights.get("rally_critiques", {}), verdict)
    with tab_video:
        _video(results)
    with tab_coach:
        _coach(report, api_key)
