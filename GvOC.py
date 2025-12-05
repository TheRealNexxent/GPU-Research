# Modern GPU vs 2000s CPU
print("=== CASE 4: Modern GPU vs 2000s CPU ===")

time_gpu = bench_gpu()
time_2000s_cpu = bench_2000s_cpu_simulated()

print(f"Modern GPU: {time_gpu:.4f}s")
print(f"2000s CPU (simulated): {time_2000s_cpu:.4f}s")
print(f"GPU Speedup: {time_2000s_cpu/time_gpu:.2f}x\n")
