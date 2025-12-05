# Modern GPU vs Modern CPU
print("=== CASE 3: Modern GPU vs Modern CPU ===")

def bench_gpu():
    N = 4000
    with tf.device('/GPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

def bench_cpu():
    N = 4000
    with tf.device('/CPU:0'):
        a = tf.random.normal((N, N), dtype=tf.float32)
        b = tf.random.normal((N, N), dtype=tf.float32)
        
        start = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        return time.time() - start

time_gpu = bench_gpu()
time_cpu = bench_cpu()

print(f"Modern GPU: {time_gpu:.4f}s")
print(f"Modern CPU: {time_cpu:.4f}s")
print(f"GPU Speedup: {time_cpu/time_gpu:.2f}x\n")
