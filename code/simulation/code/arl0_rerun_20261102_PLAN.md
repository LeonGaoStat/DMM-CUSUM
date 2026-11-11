# DMM-CUSUM IC rerun, fixed before execution

User-authorized scope: all 15 Table 3 cells: Cases I, II, III and m0=500,
1000, 1500, 2000, 3000. DMM-CUSUM only. Each cell uses 100 independent
Phase I histories, 10,000 bootstrap calibration runs per history and
10,000 independent Phase II validation runs per history. No selection by
performance, no validation-based threshold tuning, and no overwrite of old results.

## Implementation decisions

- Preserve Table 3 population and estimator: run_simulation.Config(seed=20260913),
  k=0.5, existing spectral/multiplicative mean fit and fit_spring. This isolates
  calibration and Monte Carlo design changes; it does not silently replace the
  original estimator with a different constrained optimizer.
- Per case/outer replicate generate one length-3000 stationary IC history and
  use its five prefixes. Population parameters remain fixed across outer replicates.
- Fit each prefix separately. All fitted quantities are fixed in calibration
  and validation. Do not add extra calibration observations.
- Prespecified bootstrap block length=32 observations (eight complete cycles),
  noncircular moving blocks, source start phase matched to target phase.
  This is a recorded implementation choice; the manuscript does not specify it.
- Bootstrap the preprediction standardized global deviance S^D. With fixed
  parameters and phase-aligned indices this is exactly equivalent to resampling
  complete network matrices. Do not resample innovations U or reset at block joins.
- Each bootstrap or validation path starts with both CUSUMs and spring length zero.
  Joint reset, strict >h signaling, k=0.5; prediction uses min(r,bmax).
- Record complete first-passage times. No unalarmed path is assigned a finite
  substitute run length. A numerical safety guard must be reported as failure,
  never counted as an observed alarm.
- Within each case/outer replicate the five m0 charts may share the same future
  count path (common random numbers). The 10,000 paths and different outer
  replicates use disjoint seeds. Each chart stops at its own first alarm.
- For each cell report mean of 100 history-specific means and SE=sd(means)/sqrt(100).
  Preserve all run lengths and fit/calibration metadata. Report calibration ARL,
  conditional validation estimates, control-limit distribution, and elapsed time.
- Compare with results/final/table_arl0_long.csv; do not attribute every difference
  to replication counts because old calibration used separate DGP-generated pools.

## Tasks and checks

1. Independent exact DGP/monitoring engine with common paths across five fits.
   Verify stationary latent/count moments, fixed-data recursion, identical-chart
   equality, deterministic path seeds, and no silent censoring.
2. Phase-matched bootstrap and control-limit search.
   Verify block alignment, whole-network/scalar equivalence, state recursion,
   same-seed monotonic run lengths in h, and explicit guard behavior.
3. Resumable runner with immutable source snapshot, prespecified seeds, per-history
   checkpoints and progress, and exact 100x10,000 counts in each of 15 cells.
4. Run small implementation smoke checks, then the full fixed design. Smoke
   output is never mixed with the formal result.
5. Independently inspect source and all completion counts, recompute summaries
   from archived run lengths, compare old/new results, and report limitations.

Runtime: /opt/anaconda3/bin/python, NumPy, SciPy and Numba. Existing sources and
Overleaf remain unchanged. New engine modules and results live under simulation.
