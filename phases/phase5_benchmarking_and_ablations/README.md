# Phase 5: Multi-Seed Benchmarking & Ablation Suite

This directory contains statistical multi-seed evaluation suites and comparative ablation scripts across all 5 paradigms for the course report and publication.

## Running Phase 5 Benchmarks

```bash
python phases/phase5_benchmarking_and_ablations/run_multi_seed_benchmark.py
```

### Statistical Evaluation Protocol
- 5 independent random seeds: `42, 137, 256, 1024, 2026`
- Reports mean $\pm$ standard deviation for throughput, milestone progression, unique cell discovery rate, and delta compression efficiency.
