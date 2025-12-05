# GPU Evolution Benchmark: A Comparative Analysis of GPU vs CPU Performance Across Decades

#Abbreviations used:
- G = GPU
- C = CPU
- OG = Old GPU
- OC = Old CPU
- v = Versus/Comparison

## Overview

This repository contains a comprehensive empirical study analyzing the architectural evolution and performance transition of GPUs from the 2000s era to modern accelerators. Through systematic benchmarking on Google Colab, this project quantifies how GPU technology evolved and why GPUs became the dominant computing paradigm for parallel workloads.

## Research Scope

This project investigates six key performance comparisons:

1. **Modern GPU vs 2000s GPU** – Hardware acceleration over 20+ years
2. **Modern CPU vs 2000s CPU** – Multi-core and architectural improvements
3. **Modern GPU vs Modern CPU** – Contemporary parallel computing superiority
4. **Modern GPU vs 2000s CPU** – GPU dominance against legacy single-core systems
5. **2000s GPU vs 2000s CPU** – Historical GPU-CPU dynamics
6. **2000s GPU vs Modern CPU** – Legacy GPU vs contemporary general-purpose computing

## Key Topics

- **GPU Architecture Evolution**: Memory bandwidth, core counts, SIMD/SIMT capabilities
- **Parallel Processing**: Thread prioritization, warp scheduling, memory hierarchy
- **Performance Metrics**: Matrix multiplication, vector addition, FPS-equivalent throughput analysis
- **Historical Context**: Pentium 4 (2003), GeForce 3/4 (2001-2002) vs Tesla T4, modern Xeons

## Experiments Included

### 1. Matrix Multiplication Benchmarks
- Comparative runtimes across GPU/CPU generations
- Scaling analysis with varying matrix sizes

### 2. Vector Addition Operations
- Large-scale parallelism efficiency (50M+ element arrays)
- Memory bandwidth utilization

### 3. CPU Throttling Simulation
- Single-core limitation (2000s CPU constraints)
- Thread parallelism reduction
- SIMD instruction disabling

### 4. GPU Memory Overhead Modeling
- PCIe bandwidth simulation (legacy systems)
- Data transfer bottleneck quantification

## Running on Google Colab

### Quick Start

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Set Runtime → Change runtime type → GPU (Hardware accelerator)
4. Run setup cell to verify GPU/CUDA availability
5. Execute benchmark snippets sequentially
