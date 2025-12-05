# Modern CPU vs 2000s CPU
def bench_2000s_cpu_simulated():
    """Simulate 2000s CPU constraints"""
    # Pentium 4 3.0GHz (2003): single core, no SSE3, ~0.5 IPC
    # Apply slowdown factor based on SPEC scores
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    N = 1000  # Smaller workload
    
    with tf.device('/CPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        base_time = time.time() - start
    
    # 2000s CPU ~15x slower than modern CPU (based on SPEC CPU2000 scores)
    # Pentium 4: ~800 SPECint2000, Modern Xeon: ~12000 SPECint2017
    return base_time * 15.0

def bench_modern_cpu():
    """Modern CPU with full parallelism"""
    os.environ['OMP_NUM_THREADS'] = '0'  # Use all cores
    
    N = 4000
    
    with tf.device('/CPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

print("=== CASE 2: Modern CPU vs 2000s CPU ===")
time_2000s_cpu = bench_2000s_cpu_simulated()
time_modern_cpu = bench_modern_cpu()

print(f"2000s CPU (simulated): {time_2000s_cpu:.4f}s")
print(f"Modern CPU: {time_modern_cpu:.4f}s")
print(f"Speedup (2000s→Modern): {time_2000s_cpu/time_modern_cpu:.2f}x\n")
