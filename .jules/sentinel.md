## 2024-05-23 - Duplicate Code & Resource Exhaustion (DoS)
**Vulnerability:** Detected a large block of duplicate code in `app.py`. The "live" version (executed on reruns) contained a flaw where it unconditionally created a new `tempfile.mkdtemp` directory on every interaction, leading to potential disk space exhaustion (DoS). The "dead" version (unreachable after `st.rerun()`) actually contained the correct logic to reuse the directory.
**Learning:** Copy-paste errors can lead to zombie code that hides the "correct" implementation while the active code remains vulnerable. Streamlit's execution model (reruns) makes resource management critical; always check if a resource already exists in `st.session_state` before creating a new one.
**Prevention:**
1. Avoid large blocks of duplicate code; refactor into functions.
2. When using `tempfile` in Streamlit, always cache the directory path in `st.session_state` and check for its existence before creating a new one.
3. Remove dead code immediately to prevent confusion.
