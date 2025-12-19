# Sentinel's Journal

## 2025-12-19 - Command Injection Prevention
**Vulnerability:** Detected usage of `os.system` with formatted strings constructed from file paths in `app.py`.
**Learning:** Even when inputs seem safe (e.g., from `tempfile`), using `os.system` creates a risk of command injection if the inputs are ever influenced by user data or if the file paths contain shell metacharacters. It is "Bad Security Code".
**Prevention:** Always use `subprocess.run` with a list of arguments instead of `os.system` or `subprocess.run` with `shell=True`. This prevents the shell from interpreting the arguments, mitigating command injection risks.
