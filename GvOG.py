# Modern GPU vs 2000s GPU
# 2000s GPUs: GeForce 3/4, Radeon 8500 (2001-2002)
# Limited memory (~64MB-256MB), no CUDA, lower bandwidth
# Simulate: small matrix, add artificial latency overhead

def bench_2000s_gpu_simulated():
    """Simulate 2000s GPU constraints"""
    # 2000s GPUs had extreme memory pressure
    N = 512  # Tiny by today's standards
    
    with tf.device('/GPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        # Simulate memory copy overhead (2000s had slow PCIe, no unified memory)
        start = time.time()
        for _ in range(3):  # Repeated transfers
            _ = a.numpy()
            _ = b.numpy()
        
        overhead = time.time() - start
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        compute_time = time.time() - start
    
    return compute_time + overhead * 2  # Penalize for memory overhead

def bench_modern_gpu():
    """Modern GPU (T4/V100)"""
    N = 4000
    with tf.device('/GPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

print("=== CASE 1: Modern GPU vs 2000s GPU ===")
time_2000s_gpu = bench_2000s_gpu_simulated()
time_modern_gpu = bench_modern_gpu()

print(f"2000s GPU (simulated): {time_2000s_gpu:.4f}s")
print(f"Modern GPU: {time_modern_gpu:.4f}s")
print(f"Speedup (2000s→Modern): {time_2000s_gpu/time_modern_gpu:.2f}x\n")
