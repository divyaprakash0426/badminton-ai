# Badminton-AI Improvement Roadmap

> Purpose: the reference plan for the next research and development cycle. It turns the CoachAI datasets already in the repository into (a) a real end-to-end benchmark, (b) domain-adapted models and (c) a better-grounded AI coach. It also gives the storage, compute and risk analysis needed to decide what to run.
>
> Status: proposal only. Nothing described here has been downloaded or trained yet.
> Analysis date: 2026-10-26. All numbers below were measured from repository files or YouTube metadata (`yt-dlp --skip-download -j`) unless they are marked as estimates.

---

## Table of contents
1. [Current baseline and gaps](#1-current-baseline-and-gaps)
2. [Data assets inventory](#2-data-assets-inventory)
3. [Label taxonomy alignment](#3-label-taxonomy-alignment)
4. [Recommended path (phases 0–8)](#4-recommended-path)
5. [Storage analysis](#5-storage-analysis)
6. [Compute and time budget](#6-compute-and-time-budget)
7. [Evaluation protocol and metrics](#7-evaluation-protocol-and-metrics)
8. [Proposed repository layout](#8-proposed-repository-layout)
9. [Paper plan and target venues](#9-paper-plan-and-target-venues)
10. [Risks, licensing and mitigations](#10-risks-licensing-and-mitigations)
11. [Decisions pending](#11-decisions-pending)
12. [Appendix: useful commands and facts](#12-appendix)

---

## 1. Current baseline and gaps

### 1.1 What exists (commit `294b576`)

| Component | Implementation | File(s) |
|---|---|---|
| Video I/O and concurrency | Single-pass decode feeding YOLO and TrackNet concurrently | `project_utils/parallel_processor.py`, `project_utils/video_io.py` |
| Court calibration | 4 manual corners (order TL, TR, BL, BR), mapped by homography to BWF metres | `app.py`, `engine/geometry.py` |
| Player tracking | YOLO11x-pose (TensorRT engine) assigning Near/Far players | `engine/tracker.py` (`YOLOTracker`) |
| Shuttle tracking | TrackNetV3 with InpaintNet rectification | `engine/tracker.py` (`TrackNetTracker`), `engine/tracknet_v3/` |
| Hit and rally segmentation | Rule-based (trajectory and player proximity) | `engine/analysis.py` |
| Shot classifier | SlowFast R50 (Kinetics-400 init), 18 VideoBadminton classes, 2 s or 32-frame clips | `engine/action_recognition.py`, `training/train_shot_classifier.py`, `ckpts/best_shot_classifier.pth` |
| Score and winner | `ScoreBoard` heuristic (in/out geometry plus last hitter) | `engine/analysis.py` |
| Movement statistics | Per-player distance and speed from box positions through the homography | `engine/analysis.py` (`_analyze_player_movement`) |
| AI coach | Gemini (`gemini-3.8-flash`, then `3.7-flash`, then `3.5-flash`) in JSON mode, plus follow-up chat | `engine/gemini_coach.py` |
| UI | Streamlit "Kinetic Telemetry" dashboard | `app.py`, `project_utils/ui_theme.py`, `project_utils/results_view.py` |
| Paper | IEEEtran draft, eval scripts, figures | `paper/` |

### 1.2 Measured results (paper draft)
- SlowFast on the VideoBadminton validation split (1,558 clips):
  - top-1 accuracy 86.5% (95% CI 84.7–88.2), top-5 98.3%;
  - macro-F1 0.799 (0.769–0.825), mean class accuracy 78.9%;
  - 39.9 ms per clip on the RTX 4080 Laptop.
- Weak classes: Tap Smash 0.44 F1 (recall 0.29), Transitional Slice 0.53, Push Shot 0.61, Defensive Clear 0.68.
- Full broadcast clip (`videos/sample_match.mp4`, 184 s):
  - 14 rallies and 120 shots, of which 24 are "Unknown" (20%);
  - Defensive Clear predicted 42 times, a suspected bias;
  - peak player speed 15 m/s, which is not physically plausible.
- Throughput is about 30 frames/s for YOLO plus TrackNet.
- AI coach: 26 of 27 shot terms cited in its critiques agree with the pipeline's own labels. This is consistency, not correctness.

### 1.3 Gaps that block a credible paper

| # | Gap | Consequence |
|---|---|---|
| G1 | The VideoBadminton split is clip-level: all 77 validation sessions also appear in training, and there is no test set | Accuracy is optimistic and not comparable to the dataset paper |
| G2 | No broadcast-domain training data | Domain shift: 20% Unknown and class bias |
| G3 | No "net shot" class, although it is the most common stroke in professional singles | Systematic misclassification |
| G4 | No ground truth for hits, rallies, winner, score, landing or player positions | The end-to-end system is unvalidated |
| G5 | Movement statistics are noisy | Distances and speeds are untrustworthy |
| G6 | The coach is evaluated only for internal consistency | No evidence that the advice is correct or useful |
| G7 | Manual calibration | Usability, and a source of error |

---

## 2. Data assets inventory

### 2.1 Already on disk

| Dataset | Location | Size on disk | Content |
|---|---|---|---|
| VideoBadminton (raw) | `data/VideoBadminton/` | 2.7 GB in total for `data/` | 18-class trimmed stroke clips |
| VideoBadminton split | `data/VideoBadminton_Split/{train,val}` | (included above) | 6,264 train and 1,558 val clips over 80 recording sessions; clip-level split |
| ShuttleSet annotations | `CoachAI-Projects/ShuttleSet/set/` | about 95 MB for all of `CoachAI-Projects/` | 44 matches, 104 sets, 3,683 rallies, 36,484 strokes (2018–2021) |
| ShuttleSet22 annotations | `CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set/` | (included above) | 58 matches (50 not in ShuttleSet), 140 sets, 4,944 rallies, 52,356 strokes (2022) |
| CoachAI model code | `CoachAI-Projects/*` | (included above) | Shot Influence (TF2), Stroke Forecasting (ShuttleNet, Transformer, LSTM), Movement Forecasting (DyMF, GCN, rGCN, ...), RallyNet, Strategic Environment, Badminton Environment |

Combined, the annotations cover 8,627 rallies and 88,840 strokes, with a mean of 10.3 strokes per rally. These are raw sums: the 8 overlapping matches are counted twice, so the unique total is somewhat lower. Deduplicate by `video` before building splits.

### 2.2 ShuttleSet schema (per `set{1,2,3}.csv` row = one stroke)

```
rally, ball_round, time, frame_num, roundscore_A, roundscore_B, player, server, type,
aroundhead, backhand, hit_height, hit_area, hit_x, hit_y, landing_height, landing_area,
landing_x, landing_y, lose_reason, win_reason, getpoint_player, flaw,
player_location_area, player_location_x, player_location_y,
opponent_location_area, opponent_location_x, opponent_location_y, db
```

- `frame_num`: the hit frame, computed as seconds × fps of the original upload. **Alignment must be verified per video.**
- `hit_*`, `landing_*` and `*_location_*` are image pixel coordinates. The observed landing ranges are x 80–1224 and y 135–703, consistent with 1280×720 frames.
- `player`: A is the match winner and B the loser. `match.csv` has `winner`, `loser` and `downcourt` (which player is at the bottom of the frame).
- Rally outcomes, counted over all rows:

| `lose_reason` | Translation | Count |
|---|---|---|
| 對手落地致勝 | opponent's shot landed in (winner) | 2,692 |
| 出界 | out | 2,595 |
| 掛網 | into the net | 1,939 |
| 未過網 | did not cross the net | 886 |
| 落點判斷失誤 | misjudged landing (left it, but it was in) | 220 |
| 犯規 | fault | 10 |

- `match.csv` columns: `id, video, tournament, round, year, month, day, set, duration (min), winner, loser, downcourt, url` (ShuttleSet22 has no `url`).
- `homography.csv` has a 3×3 `homography_matrix` and the four court corners in camera pixels (`upleft_*`, `upright_*`, `downleft_*`, `downright_*`). The README says it transforms "from the real-world system back to the camera system by p = H⁻¹p′". **Verify the direction numerically before use.**

### 2.3 Rally timing (important for compute)
- Rally span, measured from the first to the last annotated hit: mean 254 frames and median 191 frames (about 7–10 s).
- Total rally time across both datasets, with 3 s padding per rally, is about **31.5 hours**. The full broadcasts total about 92 hours, so rallies are about a third of the footage.

---

## 3. Label taxonomy alignment

The ShuttleSet README gives official translations. VideoBadminton class definitions come from its paper's appendix (Li et al. 2024). Proposed mapping:

| ShuttleSet (Chinese) | ShuttleSet (English) | Count (both sets) | VideoBadminton class | Confidence |
|---|---|---|---|---|
| 發短球 | short service | 5,105 | 00 Short Serve | High |
| 發長球 | long service | 1,402 | 13 Long Serve | High |
| 長球 | clear | 7,922 | 12 Clear | High |
| 殺球 | smash | 6,618 | 14 Smash | High |
| 點扣 | wrist smash | 4,019 | 03 Tap Smash | High |
| 挑球 | lob | 14,162 | 02 Lift | High |
| 防守回挑 | defensive return lob | 913 | 10 Defensive Clear | Medium |
| 防守回抽 | defensive return drive | 899 | 11 Defensive Drive | High |
| 平球 | drive | 1,494 | 15 Flat Shot | Medium |
| 小平球 | driven flight | 94 | 17 Short Flat Shot | Medium |
| 後場抽平球 | back-court drive | 1,256 | 16 Rear Court Flat Drive | High |
| 切球 | drop | 6,086 | 05 Drop Shot and/or 08 Cut | **Low**: VideoBadminton splits drop into Drop and Cut |
| 過度切球 | passive drop | 2,443 | 07 Transitional Slice | Medium |
| 推球 | push | 5,722 | 06 Push Shot | High |
| 撲球 | rush | 1,172 | 09 Rush Shot | High |
| 擋小球 | return net | 9,248 | 04 Block | Medium |
| 勾球 | cross-court net shot | 2,248 | 01 Cross Court Flight | Medium |
| **放小球** | **net shot** | **15,552** | **none (new class)** | n/a |
| 未知球種 | unknown | 2,485 | excluded | n/a |

Notes:
- Counts are the sums of the two per-dataset counts measured in this session.
- The table translates 過度切球 as "passive drop"; the CSVs use the character 過度, while the README writes 過渡.
- The Stroke Forecasting sample CSV uses a variant label, 網前球, so normalise synonyms.

### Recommended unified taxonomy (19 or 17 classes)
- **Option A, 19 classes:** the 18 VideoBadminton classes plus Net Shot. Map ShuttleSet drop (切球) to Drop Shot and keep VideoBadminton's Cut only as a VideoBadminton-side label. Training on ShuttleSet with ambiguous Drop/Cut adds label noise.
- **Option B, 17 coarse classes:** merge Drop and Cut, then add Net Shot. Cleaner across datasets and more honest. **Recommended for training and evaluation.** Fine-grained 18-class results can still be reported on VideoBadminton alone.
- Before committing to a mapping, view about 20 sampled clips per mapped pair to confirm it. This takes about 1–2 hours by hand.

---

## 4. Recommended path

The phases are ordered by value to the paper divided by effort. Phases 0–3 are the core (roughly a solid applied-journal paper). Phases 4–8 extend it toward a stronger venue.

### Phase 0: Data acquisition and alignment (prerequisite)

**Goal:** a reproducible, verified local copy of the ShuttleSet broadcasts aligned to the annotations.

Steps:
1. Write `research/data/download_shuttleset.py`:
   - read `match.csv` and deduplicate URLs (43 unique);
   - download **720p H.264, video only** with the command in §12.2;
   - write `data/shuttleset/manifest.csv` with `video_folder, youtube_id, duration, fps, width, height, filesize, sha256, status`.
2. Write `research/data/verify_alignment.py`. For each match, sample 5 rallies, render the annotated hit frames with the hit and landing pixels overlaid, and save a contact sheet. Check them by eye.
   - If there is a constant offset (re-uploaded video), estimate it by cross-correlating the TrackNet shuttle-velocity peaks with the annotated `frame_num` values, and store `frame_offset` in the manifest.
   - Check that fps matches; videos run at 25 or 30 fps.
3. Finding ShuttleSet22 videos (optional; no URLs are given): search YouTube (`yt-dlp "ytsearch5:<players> <tournament> <round> 2022"`) and check the duration against `match.csv` and the corners against `homography.csv` by projecting them onto a frame. Store only verified matches.
4. Resolve the homography direction: project the annotated corners through `H` and `H⁻¹` and see which yields court-metre values. Store a normalised `H_img2court` per match.

Deliverables: `manifest.csv`, alignment contact sheets, `H_img2court.json`.
Acceptance:
- at least 95% of sampled hits visually aligned within ±3 frames;
- corner reprojection error under 3 px.

Effort: about 1 day of work, plus download time.

### Phase 1: End-to-end benchmark of the current system (no training)

**Goal:** quantify where the existing pipeline succeeds or fails, using ground truth. This is the single most valuable experiment for the paper.

1. **Match-level split, fixed now and reused in every phase:**
   - split the 42 ShuttleSet matches into train/val/test as 28/5/9 (about 70/10/20);
   - stratify by tournament year and gender, and keep test-set players disjoint from training where feasible;
   - freeze the split in `research/splits/shuttleset_split.json`.
2. Run the pipeline on the **test matches**, either on full sets or on rally windows (annotated first hit − 3 s to last hit + 3 s). Use the annotated corners instead of manual calibration so that calibration error is factored out; run the manual version separately for the calibration study.
3. Write `research/eval/match_events.py` to match predictions to ground truth:
   - **hits:** Hungarian matching on |Δframe| ≤ k, reported for k = 2, 3 and 5 at 25 fps;
   - **rallies:** temporal IoU at 0.5 between predicted and annotated rally spans;
   - **hitter:** Near/Far versus A/B through `downcourt`. Players switch ends between sets and at 11 points in the deciding set, so derive sides per set from the annotated player locations.
4. Metrics (see §7): hit P/R/F1, rally detection F1, segmentation IoU, hitter accuracy, shot-type accuracy on matched hits (zero-shot cross-domain), landing error in metres, player-position error in metres, rally-winner accuracy, score-trajectory accuracy, and runtime.
5. Error analysis: a confusion matrix on broadcast data, and failure categories (replays or close-ups, occlusion, shuttle out of frame, fast exchanges at the net).

Deliverables: `paper/results/e2e_shuttleset_baseline.json`, figures, and an error taxonomy.
Expected outcome: a low-to-moderate baseline. That is fine, because it becomes the "before" column.
Effort: 2–3 days of work, plus about 10–15 GPU-hours (rally windows of 9 test matches).

### Phase 2: Domain-adapted shot classifier

**Goal:** a classifier that works on broadcast footage, evaluated on unseen matches.

1. **Clip extraction** (`research/data/extract_stroke_clips.py`):
   - for each annotated stroke, cut the window `[frame_num − 0.6 s, frame_num + 0.6 s]`, configurable from 0.8 to 2.0 s;
   - resize to a 256 px short side, H.264 CRF 23, no audio;
   - store at `data/shuttleset/clips/{split}/{class}/{match}_{set}_{rally}_{round}.mp4`;
   - keep the CSV metadata: hitter side, hit and landing pixels, backhand, aroundhead.
   - Optionally crop around the hitter, using the YOLO box or the annotated `player_location`, plus context. This is likely to help, because broadcast players are small.
2. **Taxonomy:** Option B from §3 (17 classes, Drop and Cut merged, plus Net Shot). Exclude 未知球種.
3. **Experiments** (each about 5–20 epochs; init from the current checkpoint or from Kinetics-400):

| ID | Training data | Evaluation | Purpose |
|---|---|---|---|
| E2.1 | Current checkpoint (VideoBadminton) | ShuttleSet test (zero-shot, mapped labels) | Domain gap |
| E2.2 | ShuttleSet train | ShuttleSet test | In-domain |
| E2.3 | VideoBadminton plus ShuttleSet | Both test sets | Joint |
| E2.4 | E2.3 with a hitter-centred crop | Both | Effect of the crop |
| E2.5 | E2.4 plus a pose stream (keypoint sequence, small GCN or Transformer, late fusion) | Both | Multimodal |
| E2.6 | E2.4 plus a shuttle-trajectory stream (TrackNet xy sequence, 1D CNN) | Both | Trajectory cues |
| E2.7 | Alternative backbone (X3D-M or VideoMAE-B) | Both | Backbone comparison |

4. **Training details:**
   - class-balanced sampling or focal loss, because Short Flat Shot has only 94 examples;
   - label smoothing 0.1;
   - AdamW or SGD with cosine schedule and AMP;
   - temporal jitter of ±0.2 s around the hit frame;
   - horizontal flip is allowed for class but must not be used if handedness or direction labels are predicted;
   - select the checkpoint on the **val** split and report on **test** once.
5. **VideoBadminton fix:** build a **session-disjoint** split (group by the recording-session prefix, for example `2022-08-30_18-00-09_dataset_set1`) and re-run evaluation, so its numbers are honest too.
6. **Confidence calibration:** temperature scaling on val; choose the "Unknown" threshold to target a precision level, and report coverage versus accuracy curves.

Acceptance (targets, not guarantees):
- macro-F1 at least 0.70 on the ShuttleSet test set;
- "Unknown" rate under 10% on broadcast footage;
- Net Shot F1 at least 0.80.

Effort: 3–5 days of work, plus 2–6 GPU-days in total.

### Phase 3: Learned hit and rally segmentation

**Goal:** replace the rule-based hit detection with a learned temporal detector.

1. **Inputs per frame:**
   - TrackNet (x, y, visibility) and its velocity and acceleration;
   - YOLO keypoints for both players, normalised to court coordinates through the homography;
   - optionally a frame-difference energy and a scene-cut flag.
2. **Labels:** the annotated hit frames as Gaussian heat targets (σ of about 1–2 frames) and the hitter side.
3. **Model:** TCN or BiGRU (about 1M parameters) over 64–128-frame windows; output per-frame hit probability plus hitter class; peak picking with NMS.
4. **Rally logic:** group hits whose gap is under T seconds, then end the rally when the shuttle disappears or lands, or on a scene cut.
5. **Broadcast handling:**
   - a shot-boundary detector (PySceneDetect or TransNetV2);
   - a "court view" classifier that drops replays and close-ups (a binary CNN on frames, labels derived from whether the annotated rallies cover the frame);
   - optionally scoreboard OCR to re-sync rally boundaries and scores.
6. Evaluate with the Phase 1 metrics and compare against the rule-based baseline.

Acceptance:
- hit F1 at least 0.85 within ±3 frames;
- rally F1 at least 0.90.

Effort: 3–4 days, plus about 0.5 GPU-day.

### Phase 4: Reliable movement analytics

1. Use the **ankle keypoints** (COCO 15 and 16) midpoint, not the box centre, as the ground-contact point; fall back to the bottom-centre of the box.
2. Smooth with a constant-velocity Kalman filter or a Savitzky–Golay filter; gate outliers by speed (over 8 m/s) and acceleration (over 15 m/s²).
3. Handle identity swaps between Near and Far using court-half priors and the side changes at set boundaries.
4. **Validate** against the annotated `player_location_x`/`y` at hit frames (converted to metres): report MAE in metres and the percentage within 0.5 m.
5. Report distance per rally, average and peak speed, and recovery-to-base time, with uncertainty.

Acceptance:
- position MAE under 0.4 m;
- peak speed under 7 m/s (elite lunge speeds are roughly 4–6 m/s).

Effort: 1–2 days.

### Phase 5: Rally outcome, point reason and shot quality

1. **Winner and reason classifier.**
   - Input: the last 3–5 strokes of a rally (type, landing, player positions, and whether the shuttle trajectory ends inside the court).
   - Output: `getpoint_player` and `lose_reason` (winner, out, net, not over, misjudge).
   - Use gradient-boosted trees first, then a small Transformer. This replaces the `ScoreBoard` heuristic.
2. **Score tracker:** cumulative winners plus badminton rules (21 points, win by 2, cap at 30, end changes). Validate against `roundscore_A`/`roundscore_B`.
3. **Shot influence:** port the CoachAI *Shot Influence* model (TensorFlow 2.0, ICDM'21 and ACM TIST 2022) to PyTorch, or re-implement it. It gives a per-shot contribution to winning the rally, the key evidence for the coach ("your third shot lost you the initiative").

Acceptance:
- winner accuracy at least 90% given perfect strokes, and at least 80% end-to-end;
- set-score exact-match rate reported.

Effort: 3–5 days.

### Phase 6: Tactical forecasting

1. **Stroke Forecasting:** ShuttleNet (AAAI 2022) predicts the next shot type and landing.
2. **Movement Forecasting:** DyMF (AAAI 2023) predicts where players move.
3. **RallyNet** (ECML PKDD 2024) imitates player behaviour. It is optional and the most research-heavy option.
4. Integration:
   - convert the pipeline's per-stroke output into the CoachAI input format, with court-normalised coordinates as in `Stroke Forecasting/data/dataset_sample.csv`;
   - train on the ShuttleSet train matches and evaluate on the test matches using **pipeline-generated** inputs as well as ground-truth inputs, which measures how perception errors propagate.
5. Tactical features for the coach:
   - opponent tendencies (for example, the distribution of their reply to your smash by zone);
   - predictability or entropy of each player's shot selection;
   - exploitable patterns.

Effort: 4–7 days.

### Phase 7: Grounded and evaluated AI coach

1. **Evidence-first prompting.** Pass only verified, quantified facts (Phases 2–6 outputs with confidence values) and require every claim to cite an evidence ID, for example `{"claim": ..., "evidence": ["rally_7.shot_5", "pattern_smash_reply"]}`. Reject or flag claims without evidence.
2. **Automatic checks** on ShuttleSet test matches, run with ground-truth inputs and with pipeline inputs:
   - claim–evidence consistency;
   - correctness of stated outcomes against `lose_reason`;
   - hallucination rate;
   - numeric accuracy (whether the stated percentages match the computed statistics).
3. **Model comparison:** Gemini Flash versus Pro versus an open model such as Qwen or Llama (local via Ollama), plus a cost and latency table.
4. **Human study:** 3–5 coaches or advanced players rate 20 reports blind on correctness, specificity and usefulness (Likert scale), comparing coach versus template baseline versus ablated coach without the tactical evidence. Report inter-rater agreement (Krippendorff's α). Obtain informed consent; institutional ethics approval may be needed depending on the journal.
5. Deterministic options: temperature 0, a fixed JSON schema, and logging of prompts and outputs for reproducibility.

Effort: 3–5 days, plus the user study.

### Phase 8: Automatic court calibration

1. Train a court-keypoint detector (heatmap HRNet, or a YOLO-pose model with 4–12 court keypoints).
   - Labels: the four corners for the 42 ShuttleSet matches (and 50 ShuttleSet22 matches) from `homography.csv`, applied to frames sampled within rally windows.
   - Augment with colour jitter, blur and small perspective changes.
2. Optionally refine with line detection (Hough transform plus a court-model fit) and RANSAC homography.
3. Metrics:
   - corner error in pixels;
   - reprojection error in metres at the court lines;
   - effect on landing and position error compared with manual calibration.
4. UI: pre-fill the four points automatically and let the user adjust them.

Acceptance: median corner error under 5 px, and a calibration failure rate under 5%.

Effort: 2–3 days, plus about 0.5 GPU-day.

### Phase dependency graph

```
Phase 0 ──► Phase 1 ──► (baseline numbers)
   │            │
   ├──► Phase 2 (classifier) ─┐
   ├──► Phase 3 (hits/rallies) ┼──► Phase 5 (outcome/influence) ──► Phase 6 (forecasting) ──► Phase 7 (coach)
   ├──► Phase 4 (movement) ───┘
   └──► Phase 8 (auto-calibration)  [independent]
```

---

## 5. Storage analysis

### 5.1 Source videos (measured from YouTube metadata; nothing downloaded)

| Item | Matches | Duration | 720p video only | 1080p video only | Audio only |
|---|---|---|---|---|---|
| ShuttleSet (measured) | 42 available of 43 unique URLs (1 unavailable: `HTBf9wFL0mk`; 44 rows, 1 duplicate URL) | 46.5 h | **25.0 GB** | 52.1 GB | 2.7 GB |
| ShuttleSet22-only (estimated; no URLs) | 50 | 45.5 h (2,727 min from `match.csv`) | about 25 GB | about 51 GB | about 2.6 GB |
| **Total** | about 92 | about 92 h | **about 50 GB** | about 103 GB | about 5 GB |

- Measured bitrates: 0.54 GB per hour at 720p and 1.12 GB per hour at 1080p. Frame rates are 25 or 30 fps.
- **Recommendation: 720p, video only.** The pipeline runs at 1280×720, ShuttleSet coordinates are in 1280×720 pixel space, and TrackNet and SlowFast downscale anyway.

### 5.2 Derived artifacts (estimates)

| Artifact | Size | Notes |
|---|---|---|
| Rally-window cuts (optional, re-encoded 720p) | about 17 GB for 31.5 h | Alternative: seek into source videos instead |
| Stroke clips, 1.2 s, 256 px short side | about 5–9 GB (88.8k clips × about 60–100 KB) | Recommended |
| Stroke clips, 2.0 s, 256 px short side | about 9–15 GB | |
| Pre-extracted JPEG frames | about 50–60 GB | **Avoid**; decode on the fly |
| Pipeline caches (pose keypoints and TrackNet per frame, `.npz`) | about 3–6 GB for 31.5 h | About 2.8M frames × (2 players × 17 keypoints × 3 + 3) float16 |
| Hit-detector training windows | under 1 GB | Built from the caches |
| Checkpoints (about 7 classifier runs × 130–350 MB, plus small models) | about 3–5 GB | Keep only the best and last |
| Court-keypoint dataset (about 20k frames, JPEG) | about 2–3 GB | |
| Overlays and figures | about 1–5 GB | Only a few matches |

### 5.3 Budget per scenario

| Scenario | Peak disk | Long-term after cleanup |
|---|---|---|
| Phase 0–1, ShuttleSet only | **about 30 GB** | about 28 GB |
| Phases 0–4, ShuttleSet only | **about 45 GB** | about 20 GB (delete sources, keep clips and caches) |
| Phases 0–8, both datasets | **about 85–95 GB** | about 30–35 GB |

**Decision taken: all data lives on the second SSD.** The data root is `/run/media/Data/badminton-ai-data` (NVMe, ext4, mounted through `/etc/fstab`, about 476 GB free before downloads). The repo's `data` entry is a symlink to it (`data -> /run/media/Data/badminton-ai-data`, git-ignored), so every `data/...` path in code and in this document resolves there. VideoBadminton and VideoBadminton_Split were moved there (copy verified with `diff -rq`, original deleted, so only one copy exists). See `data/README.md` on the SSD for the layout. With this much space, keep the source videos; deleting them after caching is no longer required.

RAM is 30 GB and there are 22 CPU threads; the GPU is an RTX 4080 Laptop with 12 GB. Plan training batches accordingly: SlowFast R50 fits batch 8–16 at 32×256² with AMP.

---

## 6. Compute and time budget (RTX 4080 Laptop, 12 GB)

| Task | Estimate | Notes |
|---|---|---|
| Download ShuttleSet at 720p (25 GB) | 1–4 h | Depends on throttling; use `--concurrent-fragments 4` |
| Download ShuttleSet22 (search, verify) | 2–6 h | Partly manual |
| Pipeline on full ShuttleSet (46.5 h of video) | about 40–55 h | About 30 fps throughput |
| Pipeline on rally windows only (about 16 h for ShuttleSet) | **about 14–18 h** | Recommended |
| Pipeline on test-set rally windows only (9 matches) | about 3–4 h | Enough for Phase 1 |
| Clip extraction (88.8k clips, ffmpeg, 16 processes) | 1–2 h | CPU-bound |
| SlowFast fine-tune, 17 classes, about 60k training clips, 10 epochs | about 12–24 h per run | Data-loader-bound; use decord, PyAV or a DALI-style loader |
| X3D-M or VideoMAE-B comparison | about 1–2 days per run | VideoMAE-B may need gradient accumulation |
| Hit-detector TCN | under 2 h | Small |
| Winner/reason and influence models | under 2 h each | Tabular and sequence models |
| ShuttleNet and DyMF | 1–4 h each | Small datasets |
| Court-keypoint model | 4–8 h | |
| **Total GPU time (core phases 0–3)** | **about 3–6 days** | |
| **Total GPU time (all phases)** | **about 6–10 days** | |

---

## 7. Evaluation protocol and metrics

General rules:
- **Split by match, never by clip.** Freeze the split file and report on test exactly once per final model.
- Report 95% bootstrap confidence intervals, resampling **by match or rally**, not by clip.
- Fix seeds and log configurations; store all results as JSON in `paper/results/`.

| Task | Primary metric | Secondary metrics |
|---|---|---|
| Shot classification | Macro-F1 | Top-1 and top-5 accuracy, per-class F1, coverage versus accuracy, ECE (calibration) |
| Hit detection | F1 within ±3 frames | F1 within ±2 and ±5 frames, mean \|Δframe\| |
| Rally segmentation | Rally F1 at IoU 0.5 | Mean IoU, over- and under-segmentation counts |
| Hitter attribution | Accuracy | n/a |
| Landing point | Median error in metres | Percentage within 0.5 m and 1.0 m, in/out accuracy |
| Player position | MAE in metres | Percentage within 0.5 m |
| Rally winner and reason | Accuracy, macro-F1 | Confusion matrix |
| Score tracking | Exact set-score match | Mean absolute score error |
| Forecasting (ShuttleNet, DyMF) | Shot-type cross-entropy and accuracy, landing MAE | Results with ground-truth versus pipeline inputs |
| Coach | Claim-evidence consistency, hallucination rate | Human Likert scores, α, latency and cost |
| Court calibration | Median corner error in pixels | Reprojection error in metres, failure rate |
| System | Frames per second, end-to-end minutes per match | GPU memory |

Ablations worth reporting:
- TrackNet with versus without InpaintNet;
- rule-based versus learned hits;
- box centre versus ankle keypoints;
- with versus without Kalman smoothing;
- VideoBadminton-only versus joint training;
- crop versus full frame;
- pose and trajectory streams;
- coach with versus without tactical evidence.

---

## 8. Proposed repository layout

```
research/
  data/
    download_shuttleset.py      # Phase 0
    verify_alignment.py         # Phase 0
    find_shuttleset22.py        # Phase 0 (optional)
    extract_stroke_clips.py     # Phase 2
    build_hit_windows.py        # Phase 3
    taxonomy.py                 # unified label mapping (single source of truth)
  splits/
    shuttleset_split.json       # frozen match-level split
    videobadminton_session_split.json
  eval/
    match_events.py             # prediction-to-ground-truth matching
    e2e_benchmark.py            # Phase 1 runner
    metrics.py
  models/
    hit_detector.py             # Phase 3
    outcome_model.py            # Phase 5
    court_keypoints.py          # Phase 8
  train/
    train_classifier.py         # generalised from training/train_shot_classifier.py
    train_hit_detector.py
  configs/*.yaml
data/                           # git-ignored
  shuttleset/{videos,manifest.csv,clips,cache}/
  shuttleset22/...
paper/                          # results/*.json, figures/*.png, main.tex
docs/improvement-roadmap.md     # this file
```

Engineering notes:
- Keep `engine/` as the production inference path. Promote research models into `engine/` only after they beat the baseline on the frozen test split.
- Add regression tests in `tests/` for `taxonomy.py`, `match_events.py` and the homography direction.
- Store model metadata (taxonomy version, split hash, metrics) next to each checkpoint.
- Add a `requirements-research.txt` for `yt-dlp`, `decord` or `av`, `scenedetect`, `scikit-learn` and `lightgbm`. TensorFlow is only needed if Shot Influence is used unported.

---

## 9. Paper plan and target venues

**Updated contribution statement (after phases 0–3, optionally 4–7):**
1. The first (to our knowledge; verify) end-to-end evaluation of a broadcast badminton analytics pipeline against stroke-level ground truth (hits, rallies, shot types, landings, winners).
2. A cross-domain study of shot recognition from VideoBadminton to broadcast footage (ShuttleSet), with a unified taxonomy and match-disjoint splits.
3. A learned hit and rally segmenter and calibrated movement analytics.
4. Optionally, an evidence-grounded LLM coach with an automatic and human evaluation.

Suggested structure:
- Introduction
- Related Work
- Data and Taxonomy
- System
- Experiments: E2E baseline; classifier domain adaptation; segmentation; movement; outcome; coach
- Ablations
- Limitations
- Conclusion

Venue fit (mid-tier, applied):

| Scope | Venues |
|---|---|
| Phases 0–3 | *Sensors* (MDPI), *Applied Sciences*, *IEEE Access*, *Multimedia Tools and Applications*, *Journal of Imaging*; sports venues such as the *International Journal of Computer Science in Sport* and the *Journal of Sports Sciences* (more domain-oriented) |
| Phases 0–7 | Stronger options: *IEEE Transactions on Multimedia* (stretch), *Computer Vision and Image Understanding*, *Expert Systems with Applications*; workshops: CVPR CVsports, ACM MMSports |

---

## 10. Risks, licensing and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| YouTube videos removed or region-blocked | Medium (1 of 43 already gone) | Fewer matches | Download early, record the manifest, accept about 40 matches |
| Frame misalignment (re-encoded uploads, different fps) | Medium | Corrupt labels | Phase 0 alignment check, per-video offset, drop mismatched videos |
| ShuttleSet22 videos not findable | High | Smaller dataset | ShuttleSet alone (36k strokes) is sufficient for phases 0–4 |
| Taxonomy mismatch (Drop/Cut, Block/Return net) | High | Label noise | Option B coarse taxonomy, manual spot checks, report both |
| Replays or close-ups confuse the pipeline | High | False hits | Court-view classifier, scene-cut detection |
| Data-loader bottleneck in video training | High | Slow training | Short 256 px clips, decord, multiple workers, cached tensors for small experiments |
| GPU memory (12 GB) | Medium | Smaller batches | AMP, gradient accumulation, smaller models (X3D) |
| Disk usage | Low to medium | Stalls | Rally-window processing, delete sources after caching |
| LLM non-determinism and cost | Medium | Irreproducible results | Temperature 0, cached responses, local open-model comparison |
| **Licensing** | n/a | Legal and publication risk | ShuttleSet, ShuttleSet22 and VideoBadminton are research datasets (see each licence and paper); videos are BWF broadcasts on YouTube. Use them for non-commercial research only, **do not redistribute** videos or clips, publish only annotations-derived code and splits, and cite all sources. The commercial product must not ship models trained on these data without checking the licence terms. |
| Personal data (athletes' likeness) | Low | Ethics review | Publicly broadcast professional matches; avoid publishing identifiable crops beyond what is necessary |

---

## 11. Decisions pending

1. **Dataset scope:** ShuttleSet only (recommended first; 25 GB, verified URLs) or ShuttleSet plus ShuttleSet22 (about 50 GB, manual video search).
2. **Resolution:** 720p video only (recommended) or 1080p.
3. **Phase order:** recommended 0, then 1, then 2, then 3 (core), then 4, then 5, then 7, then 6, then 8.
4. **Taxonomy:** Option B with 17 coarse classes (recommended) or Option A with 19 classes.
5. **Coach study:** whether to recruit coaches or players for a human evaluation, and whether the target journal needs ethics approval.
6. **Product versus research:** whether research-trained models may be used in the production app (licensing).

---

## 12. Appendix

### 12.1 Commands used for this analysis (reproducible)

```bash
# Durations and match counts
python3 -c "import csv; r=list(csv.DictReader(open('CoachAI-Projects/ShuttleSet/set/match.csv'))); print(len(r), sum(float(x['duration']) for x in r))"

# Size and availability probe (metadata only, no download)
yt-dlp --skip-download -j "<url>"   # read formats[].filesize / filesize_approx for height 720/1080

# Free disk
df -h .
```

### 12.2 Download command (used)

```bash
cd data/shuttleset            # or data/shuttleset22
yt-dlp -f "136/bv*[height<=720][vcodec^=avc1][protocol=https]/bv*[height<=720][protocol=https]" \
  --concurrent-fragments 4 --download-archive archive.txt --no-overwrites --continue \
  --ignore-errors --write-info-json --no-progress -o "videos/%(id)s.%(ext)s" -a urls.txt
```

Notes from the actual run:
- Format 136 is H.264 1280×720 video only, about 300 MB per match. It is pinned because yt-dlp otherwise prefers AV1, which is risky to decode with OpenCV and decord; the HLS format 232 is about 5× larger.
- An outdated yt-dlp returned HTTP 403 for every video. `pipx upgrade yt-dlp` fixed it.
- ShuttleSet22 has no URLs. `research/data/find_shuttleset22.py` searches YouTube and auto-selects only strict BWF TV matches (both surnames, tournament and year in the title, video 1–2× the annotated duration). The remaining matches are listed in `data/shuttleset22/candidates.csv` for manual review. Every picked video **must** still pass the Phase 0 alignment check (court corners against `homography.csv`, stroke frames against `frame_num`) before use.

### 12.3 Key facts at a glance

| Fact | Value |
|---|---|
| ShuttleSet | 44 matches, 3,683 rallies, 36,484 strokes, 43 unique URLs, 42 available |
| ShuttleSet22 | 58 matches (50 new), 4,944 rallies, 52,356 strokes, no URLs |
| Combined | 8,627 rallies, 88,840 strokes, mean 10.3 strokes per rally |
| Rally time | Median 191 frames (about 7.6 s), about 31.5 h with padding |
| Full footage | about 92 h, about 50 GB at 720p |
| Most frequent stroke | Net shot 放小球 (15,552); missing from VideoBadminton |
| Coordinate space | 1280×720 image pixels; homography in `homography.csv` (verify direction) |
| Data root | `/run/media/Data/badminton-ai-data` (second SSD, about 476 GB free), symlinked as `data` |
| GPU, RAM and CPU | RTX 4080 Laptop 12 GB, 30 GB RAM, 22 threads |

### 12.4 References for the CoachAI components
- **ShuttleSet:** Wang et al., KDD 2023 (arXiv:2306.04948).
- **ShuttleSet22, CoachAI Challenge:** IJCAI 2023 (see `CoachAI-Projects/CoachAI-Challenge-IJCAI2023/README.md`).
- **Shot Influence:** ICDM 2021; ACM TIST 2022.
- **ShuttleNet (Stroke Forecasting):** AAAI 2022.
- **DyMF (Movement Forecasting):** AAAI 2023 (arXiv:2211.12217).
- **RallyNet:** ECML PKDD 2024 (arXiv:2403.12406).
- **VideoBadminton:** Li et al. 2024 (arXiv:2403.12385).
- **BibTeX:** `CoachAI-Projects/CITATIONS.bib`. Verify every entry before citing.
