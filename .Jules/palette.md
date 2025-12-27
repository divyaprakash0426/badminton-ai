## 2024-05-23 - Empty State Structure
**Learning:** Breaking text-heavy "Welcome" screens into 3 columns (e.g., Upload -> Calibrate -> Analyze) significantly improves readability and guiding power in Streamlit apps compared to a simple markdown list.
**Action:** Use `st.columns(3)` combined with `st.container(border=True)` for future onboarding flows to create distinct, digestible steps.
## 2025-12-22 - Progress Feedback
**Learning:** For multi-step, long-running processes (like video analysis), replacing static log messages with `st.status` containers keeps the UI clean and provides a satisfying 'complete' state.
**Action:** Wrap blocking processing logic in `with st.status('Processing...', expanded=True):` and update steps with `st.write`.
## 2025-12-22 - Toast Notifications for Reruns
**Learning:** When using `st.rerun()`, standard `st.success` messages are cleared immediately. Using a session state flag combined with `st.toast` at the top of the script provides a persistent, delightful confirmation message that survives the page reload.
**Action:** Use the `processing_complete` flag pattern + `st.toast` for all long-running processes that trigger a rerun.
