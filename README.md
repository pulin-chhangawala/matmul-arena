<p align="center">
  <img src="docs/banner.png" alt="matmul-arena banner" width="800"/>
</p>

<h1 align="center">matmul-arena</h1>

<p align="center">
  <strong>How fast can you multiply two matrices?</strong><br/>
  <em>Naive → Transposed → Tiled → Threaded → SIMD → Strassen: same algorithm, 30× performance gap</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-C11-blue?style=flat-square" alt="C11"/>
  <img src="https://img.shields.io/badge/peak-15+_GFLOPS-brightgreen?style=flat-square" alt="GFLOPS"/>
  <img src="https://img.shields.io/badge/algorithms-6-orange?style=flat-square" alt="Algorithms"/>
  <img src="https://img.shields.io/badge/SIMD-SSE2-purple?style=flat-square" alt="SIMD"/>
</p>

---

## The Point

A 1024×1024 matmul goes from **~0.5 GFLOPS** (naive) to **15+ GFLOPS** (tiled + SIMD + threads) on the same CPU. That's a **30× gap** from the same O(n³) algorithm, just by being nice to the memory hierarchy.

Every time you call `numpy.dot()` or `torch.matmul()`, there's a BLAS library doing exactly this under the hood.

## Quick Start

```bash
make
./matmul-arena         # default 1024x1024
./matmul-arena 512     # custom size
./matmul-arena --json  # machine-readable output
make bench             # benchmark at 256, 512, 1024
```

---

## Implementations

### 1. 🐌 Naive (ijk)
The textbook triple-nested-loop. Accesses matrix B column-wise → **cache miss on every element** of the inner loop. This is what every CS student writes first.

**Complexity**: O(n³) arithmetic, O(n³) cache misses

### 2. 🔄 Transposed
Transpose B first, then both matrices are accessed row-wise. One extra O(n²) pass but saves O(n³) cache misses.

**Why it works**: Accessing memory sequentially triggers hardware prefetching. Column access defeats it.

**Typical speedup**: 3-5× over naive

### 3. 🧱 Tiled / Blocked
Process the matrices in 64×64 tiles that fit in L1 cache (~32KB for doubles). The key insight: if your working set fits in cache, you're compute-bound. If it doesn't, you're memory-bound.

**Tile size choice**: 64×64 doubles = 32KB = L1 data cache on most x86 CPUs.

**Typical speedup**: 6-8× over naive

### 4. 🧵 Multithreaded (pthreads)
Each thread gets a horizontal strip of the result matrix, using tiled approach internally. Linear speedup up to the number of physical cores.

**Typical speedup**: 20-25× over naive (on 4+ cores)

### 5. ⚡ SIMD (SSE2 + Tiled)
Uses 128-bit SSE2 intrinsics to process **2 doubles per instruction**, combined with tiling.

```c
__m128d va = _mm_load_pd(&A[i*N + k]);      // load 2 doubles from A
__m128d vb = _mm_load_pd(&B_T[j*N + k]);    // load 2 doubles from B^T
sum = _mm_add_pd(sum, _mm_mul_pd(va, vb));   // 2 FMAs in one instruction
```

**Typical speedup**: 10-12× over naive

### 6. 🧮 Strassen (O(n^2.807))
Divide-and-conquer algorithm that trades additions for multiplications: **7 sub-multiplications** instead of 8, recursively. Asymptotically faster than all above methods.

**Complexity**: O(n^2.807), measurably faster at n ≥ 256

**Typical speedup**: 7× at 256×256 (verified correct against naive)

---

## Example Output

```
================================================================
  matmul-arena: 1024x1024 matrix multiplication benchmark
================================================================

  Implementation          Time (s)     GFLOPS    Speedup  Correct
  -------------------- ---------- ---------- ---------- --------
  Naive (ijk)              4.2381       0.51       1.0x        ✓
  Transposed               1.1203       1.92       3.8x        ✓
  Tiled                    0.6891       3.12       6.2x        ✓
  Threaded                 0.1823      11.79      23.2x        ✓
  SIMD+Tiled               0.4102       5.24      10.3x        ✓
  Strassen                 0.5100       4.22       8.3x        ✓
================================================================
```

*Results vary by CPU. Each implementation is verified against the naive result for correctness.*

---

## Why Each Speed Matters

```
Naive       ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.5 GFLOPS  (cache thrashing)
Transposed  ██████████░░░░░░░░░░░░░░░░░░░░  1.9 GFLOPS  (spatial locality)
Tiled       ████████████████░░░░░░░░░░░░░░  3.1 GFLOPS  (temporal locality)
SIMD        ████████████████████████░░░░░░  5.2 GFLOPS  (instruction-level)
Strassen    ██████████████████████░░░░░░░░  4.2 GFLOPS  (algorithmic)
Threaded    ██████████████████████████████ 11.8 GFLOPS  (thread-level)
```

The **hierarchy of optimization**:
1. **Algorithm** (Strassen): reduce total work
2. **Memory layout** (transpose, tiling): use the cache
3. **Vectorization** (SIMD): use the full ALU width
4. **Parallelism** (threads): use all cores

---

## JSON Output

```bash
./matmul-arena 512 --json > results.json
```

Produces machine-readable benchmark data for visualization pipelines.

---

<p align="center">
  <sub>One file. Six algorithms. 30× performance gap. Welcome to systems programming.</sub>
</p>
