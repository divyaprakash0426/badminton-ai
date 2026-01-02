## 2024-05-23 - Empty State Structure
**Learning:** Breaking text-heavy "Welcome" screens into 3 columns (e.g., Upload -> Calibrate -> Analyze) significantly improves readability and guiding power in Streamlit apps compared to a simple markdown list.
**Action:** Use `st.columns(3)` combined with `st.container(border=True)` for future onboarding flows to create distinct, digestible steps.
## 2025-12-22 - Progress Feedback
**Learning:** For multi-step, long-running processes (like video analysis), replacing static log messages with `st.status` containers keeps the UI clean and provides a satisfying 'complete' state.
**Action:** Wrap blocking processing logic in `with st.status('Processing...', expanded=True):` and update steps with `st.write`.
## 2025-12-22 - Toast Notifications for Reruns
**Learning:** When using `st.rerun()`, standard `st.success` messages are cleared immediately. Using a session state flag combined with `st.toast` at the top of the script provides a persistent, delightful confirmation message that survives the page reload.
**Action:** Use the `processing_complete` flag pattern + `st.toast` for all long-running processes that trigger a rerun.
## 2024-05-24 - Data Presentation
**Learning:** Displaying raw frame numbers in tables is helpful for debugging but alienating for users. Converting frames to timestamps (MM:SS) makes the data instantly relatable and usable for analysis.
**Action:** Always provide a "Time" column in data tables derived from video analysis, calculating `frame / fps`.
## 2024-05-24 - Navigation Precision
**Learning:** For timeline-based tasks like selecting a specific frame, standard sliders and coarse increments (1s) are often insufficient. Users struggle to find the "perfect" frame. Adding "Fine" (+/- 1 frame) vs "Coarse" (+/- 1 second) controls dramatically reduces user frustration.
**Action:** When implementing time/frame selectors, always provide a "Step" and a "Micro-Step" control option.
