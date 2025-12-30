## 2024-05-23 - Duplicate Code & Resource Exhaustion (DoS)
**Vulnerability:** Detected a large block of duplicate code in `app.py`. The "live" version (executed on reruns) contained a flaw where it unconditionally created a new `tempfile.mkdtemp` directory on every interaction, leading to potential disk space exhaustion (DoS). The "dead" version (unreachable after `st.rerun()`) actually contained the correct logic to reuse the directory.
**Learning:** Copy-paste errors can lead to zombie code that hides the "correct" implementation while the active code remains vulnerable. Streamlit's execution model (reruns) makes resource management critical; always check if a resource already exists in `st.session_state` before creating a new one.
**Prevention:**
1. Avoid large blocks of duplicate code; refactor into functions.
2. When using `tempfile` in Streamlit, always cache the directory path in `st.session_state` and check for its existence before creating a new one.
3. Remove dead code immediately to prevent confusion.
## 2025-10-26 - Insecure Deserialization in TrackNet
**Vulnerability:** Found multiple instances of `torch.load` without `weights_only=True` in the `engine/tracknet_v3` module. This allows loading arbitrary pickled objects, potentially leading to Remote Code Execution (RCE) if a user loads a malicious checkpoint.
**Learning:** Even if the main app logic is secure, utility scripts and training code in the repo can expose users to risk. Security patches must cover the entire codebase, not just the "hot path".
**Prevention:** Always use `torch.load(..., weights_only=True)` when loading model weights. This restricts unpickling to safe types (tensors, primitives). This feature is available in PyTorch 2.4+ (and earlier versions via other means, but `weights_only` is the modern standard).
