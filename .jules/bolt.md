## 2024-05-23 - Camera Cut Detection Optimization
**Learning:** For global scene change detection (camera cuts), downsampling the frame to a small fixed size (e.g., 64x64) is sufficient and significantly faster than using a relative scale (e.g., 0.25x).
**Action:** When implementing global image statistics calculations (histograms, average color), always resize to a small fixed thumbnail first to reduce pixel count and ensuring consistent performance regardless of input resolution.

## 2024-05-24 - ThreadPoolExecutor Overhead for Small Tasks
**Learning:** For extremely lightweight tasks (e.g., small 64x64 image histogram calculation taking < 0.2ms), the overhead of `ThreadPoolExecutor` (context switching, task submission) can outweigh the benefits of parallelization, even in CPU-bound scenarios.
**Action:** Benchmark parallel vs sequential implementations for small batch tasks. If the per-item processing time is negligible, prefer sequential list comprehensions over `executor.map` to avoid thread management overhead.

## 2024-05-25 - GPU Vectorized Thresholding
**Learning:** Performing binary thresholding directly on the GPU (e.g., `(tensor > 0.5).to(torch.uint8)`) prior to CPU transfer reduces data transfer size by 4x (float32 to uint8) and leverages parallel processing, significantly outperforming float32 transfers followed by CPU-side loop thresholding.
**Action:** When working with model outputs that require thresholding, perform the threshold and type conversion on the device (GPU) before moving data to the CPU.
