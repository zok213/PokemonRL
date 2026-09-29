# Phase 5: Multi-Seed Benchmarking & Ablation Suite

This directory contains statistical multi-seed evaluation suites and comparative ablation scripts across all paradigms for the course report and publication.

## Scripts & Evaluations

### 1. Head-to-Head Comparison: Phase 1 Baseline vs Phase 3 Warm-Started Agent
Direct side-by-side comparison between the Pleines et al. cold-start baseline and our upgraded warm-started agent:
```bash
python phases/phase5_benchmarking_and_ablations/compare_phase1_vs_phase3.py
```
- Measures visual feature separation (2.64x contrast improvement).
- Verifies Nurse Joy healing trap immunity (100% exploit in baseline vs 0% in upgraded agent).
- Compares milestone progression depths across all 8 badges and Safari Zone.
- Quantifies active parameter memory savings (79.1% trainable parameter reduction).

### 2. Multi-Seed Statistical Benchmark Suite
Evaluates throughput, cell discovery, and delta compression across 5 independent seeds:
```bash
python phases/phase5_benchmarking_and_ablations/run_multi_seed_benchmark.py
```

### Statistical Evaluation Protocol
- 5 independent random seeds: `42, 137, 256, 1024, 2026`
- Reports mean $\pm$ standard deviation for throughput, milestone progression, unique cell discovery rate, and delta compression efficiency.
