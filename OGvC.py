# 2000s GPU vs Modern CPU
print("=== CASE 6: 2000s GPU vs Modern CPU ===")

time_2000s_gpu = bench_2000s_gpu_simulated()
time_modern_cpu = bench_modern_cpu()

print(f"2000s GPU (simulated): {time_2000s_gpu:.4f}s")
print(f"Modern CPU: {time_modern_cpu:.4f}s")
print(f"Modern CPU Speedup: {time_2000s_gpu/time_modern_cpu:.2f}x\n")
