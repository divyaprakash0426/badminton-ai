import numpy as np
import time

H, W = 288, 512
C = 3
SEQ_LEN = 3
BATCH_SIZE = 16

# Create dummy frames (simulating extended_frames)
# We need BATCH_SIZE + SEQ_LEN - 1 frames
TOTAL_FRAMES = BATCH_SIZE + SEQ_LEN - 1
frames = [np.random.randn(C, H, W).astype(np.float32) for _ in range(TOTAL_FRAMES)]

def original_method(frames, batch_size, seq_len):
    batch_inputs = []
    for i in range(batch_size):
        window = frames[i : i+seq_len]
        input_tensor = np.concatenate(window, axis=0)
        batch_inputs.append(input_tensor)
    return np.stack(batch_inputs)

def optimized_method(frames, batch_size, seq_len):
    # Output shape: (B, C*seq_len, H, W)
    out = np.empty((batch_size, C*seq_len, H, W), dtype=np.float32)
    for i in range(batch_size):
        for j in range(seq_len):
            out[i, j*3 : (j+1)*3] = frames[i+j]
    return out

# Warmup
print("Warmup...")
res1 = original_method(frames, BATCH_SIZE, SEQ_LEN)
res2 = optimized_method(frames, BATCH_SIZE, SEQ_LEN)

# Verify
print(f"Match: {np.allclose(res1, res2)}")

# Benchmark
print("Benchmarking...")
start = time.time()
ITER = 50
for _ in range(ITER):
    original_method(frames, BATCH_SIZE, SEQ_LEN)
end = time.time()
print(f"Original: {(end-start)/ITER:.5f}s")

start = time.time()
for _ in range(ITER):
    optimized_method(frames, BATCH_SIZE, SEQ_LEN)
end = time.time()
print(f"Optimized: {(end-start)/ITER:.5f}s")
