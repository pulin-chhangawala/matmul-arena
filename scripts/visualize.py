"""
visualize.py - Performance visualization for matmul benchmarks

Generates charts comparing implementations and a roofline model
to show where each implementation sits relative to the hardware's
theoretical peak.

Usage:
    python scripts/visualize.py results.json
"""

import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(results, outpath='results/comparison.png'):
    """Bar chart comparing GFLOPS across implementations."""
    names = [r['name'] for r in results]
    gflops = [r['gflops'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    
    # GFLOPS
    bars = ax1.barh(names, gflops, color=colors)
    ax1.set_xlabel('GFLOPS')
    ax1.set_title('Performance (higher = better)')
    for bar, val in zip(bars, gflops):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    
    # Speedup
    bars = ax2.barh(names, speedups, color=colors)
    ax2.set_xlabel('Speedup vs Naive')
    ax2.set_title('Relative Speedup')
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}x', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'  Saved: {outpath}')


def plot_roofline(results, peak_gflops=None, peak_bandwidth_gb=None,
                  outpath='results/roofline.png'):
    """
    Roofline model plot.
    
    The roofline model shows the relationship between computational 
    intensity (FLOPS/byte) and achievable performance. Implementations
    below the roofline are either memory-bound or compute-bound.
    
    For matmul (N^3 operations, 3*N^2*8 bytes of data for doubles):
      operational intensity ≈ N/24 FLOPS/byte
    """
    if peak_gflops is None:
        peak_gflops = max(r['gflops'] for r in results) * 1.5
    if peak_bandwidth_gb is None:
        peak_bandwidth_gb = 25.0  # typical DDR4
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # roofline
    x = np.logspace(-1, 3, 200)
    memory_roof = peak_bandwidth_gb * x
    compute_roof = np.full_like(x, peak_gflops)
    roof = np.minimum(memory_roof, compute_roof)
    
    ax.loglog(x, roof, 'k-', linewidth=2, label='Roofline')
    ax.fill_between(x, roof, alpha=0.1, color='gray')
    
    # plot each implementation
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(results)))
    
    n = results[0].get('matrix_size', 1024)
    op_intensity = n / 24.0  # FLOPS per byte for NxN matmul
    
    for i, r in enumerate(results):
        ax.plot(op_intensity, r['gflops'], markers[i % len(markers)],
               color=colors[i], markersize=10, label=r['name'])
    
    ax.set_xlabel('Operational Intensity (FLOPS/byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title(f'Roofline Model (N={n})', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 1000)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'  Saved: {outpath}')


def plot_scaling(sizes, gflops_per_impl, names, outpath='results/scaling.png'):
    """Plot how each implementation scales with matrix size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, gflops in zip(names, gflops_per_impl):
        ax.plot(sizes, gflops, 'o-', label=name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12)
    ax.set_ylabel('GFLOPS', fontsize=12)
    ax.set_title('Performance Scaling', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f'  Saved: {outpath}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results.json>")
        print("\nTo generate results.json, run matmul-arena with --json flag")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        data = json.load(f)
    
    import os
    os.makedirs('results', exist_ok=True)
    
    plot_comparison(data['results'])
    plot_roofline(data['results'])
