# Leg-arm correlation analysis

Task: ee_walk; mode: configured; seeds: [11, 22, 33, 44, 55]; steps/seed: 600; segments/seed: 3; horizon index: 0.

Simulation warnings: 0. Low ESS fraction (all iterations): 98.7%. ESS flag: ESS < max(20.0, 0.2 × batch samples).

Final-iteration low ESS fraction: 99.2%. Reliable coverage across seed/segment groups: 0.0% to 8.0%.

Candidate pairs: 0. Insufficient reliable coverage; do not infer repeatability.

Maximum absolute group-mean weighted correlation: 0.10829822782427177. Maximum absolute paired weighted-minus-uniform mean: 0.1052948234899331.

Screen: at least 5 seeds, 3 segments, >=80% reliable updates and >=20 reliable updates in every seed/segment; |mean weighted r| >=0.15, |mean weighted r - mean uniform r| >=0.10, and same sign in >=80% of seed/segment means. These are exploratory thresholds, not significance tests. CSV confidence intervals use 2000 whole-seed bootstrap resamples (pointwise 95%, no multiplicity correction; few seeds limit precision). Seeds and segments are equally weighted. Low-ESS updates are excluded from repeatability, retained in raw data and viewer. Fixed/zero-variance axes are NaN.

This measures post-clipping action proposals conditioned on cost, not causal coupling. No horizon pooling. All horizons/iterations and weighted cross-covariance are stored in seed NPZ files; viewer and summary use only the final iteration at the specified horizon. Low ESS flags uncertainty in correlation estimation; it does not by itself imply poor control performance. Constant or low-ESS samples cannot support reliable estimates. Check simulation warnings and final state in seed JSON files. Separate phase/command-conditioned analyses and longer independent runs are needed before generalizing; cancellation across gait phases can conceal coupling. Overlapping horizons/adjacent updates are not independent observations. No p-values are claimed across the 72 pairs. Improvement in task performance requires a separate controlled intervention; this analysis does not modify sampling.
