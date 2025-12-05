# 2000s GPU vs 2000s CPU
print("=== CASE 5: 2000s GPU vs 2000s CPU ===")

time_2000s_gpu = bench_2000s_gpu_simulated()
time_2000s_cpu = bench_2000s_cpu_simulated()

print(f"2000s GPU (simulated): {time_2000s_gpu:.4f}s")
print(f"2000s CPU (simulated): {time_2000s_cpu:.4f}s")

if time_2000s_gpu < time_2000s_cpu:
    print(f"GPU Speedup: {time_2000s_cpu/time_2000s_gpu:.2f}x")
else:
    print(f"CPU was comparable/faster (limited GPU memory in 2000s)")
print()
