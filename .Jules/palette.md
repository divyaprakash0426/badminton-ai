## 2024-05-23 - Empty State Structure
**Learning:** Breaking text-heavy "Welcome" screens into 3 columns (e.g., Upload -> Calibrate -> Analyze) significantly improves readability and guiding power in Streamlit apps compared to a simple markdown list.
**Action:** Use `st.columns(3)` combined with `st.container(border=True)` for future onboarding flows to create distinct, digestible steps.
