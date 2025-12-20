## 2024-05-23 - Camera Cut Detection Optimization
**Learning:** For global scene change detection (camera cuts), downsampling the frame to a small fixed size (e.g., 64x64) is sufficient and significantly faster than using a relative scale (e.g., 0.25x).
**Action:** When implementing global image statistics calculations (histograms, average color), always resize to a small fixed thumbnail first to reduce pixel count and ensuring consistent performance regardless of input resolution.
