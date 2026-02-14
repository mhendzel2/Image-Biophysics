# Advanced SPT Upgrade Roadmap

## Objective

Upgrade the SPT stack from MSD-only and proxy state inference to a bias-aware, uncertainty-aware workflow suitable for kinetic claims.

## Implemented in This Revision

1. **Bias-aware diffusion inference (Spot-On style)**
- New module: `spt_models/spot_on.py`
- Corrections included:
  - Localization error in displacement emissions
  - Motion blur correction via effective lag time (`dt - exposure/3`)
  - Axial defocalization survival weighting
  - Tracking max-step truncation correction
- Output includes:
  - Diffusion coefficients and state fractions
  - AIC/BIC and log-likelihood
  - Bootstrap confidence intervals

2. **True switching diffusion HMM**
- New module: `spt_models/switching_diffusion.py`
- Implements:
  - Forward-backward posterior inference
  - Baum-Welch (EM) parameter learning
  - Viterbi decoding
  - Transition matrix and dwell-time outputs
  - Bootstrap uncertainty intervals
- Emissions include localization error, motion blur, and optional truncation.

3. **Bayesian posterior workflow**
- New module: `spt_models/bayesian.py`
- Implements:
  - Metropolis-Hastings posterior sampling over Spot-On-like likelihood
  - Credible intervals for D and fractions
  - Acceptance and ESS diagnostics
  - Optional external backend detection (`bayes_traj`) with internal fallback

4. **Representation learning foundations**
- New module: `spt_models/trajectory_representation.py`
- Includes:
  - Synthetic trajectory generator with domain randomization:
    - Brownian, confined, anomalous, directed, binding/unbinding
  - Feature embedding baseline
  - Optional transformer encoder scaffold (PyTorch if available)

5. **Benchmark/metrics utilities**
- New module: `spt_models/benchmarking.py`
- Metrics implemented:
  - Diffusion bias/relative error
  - Fraction recovery error
  - Transition matrix error
  - Dwell-time extraction from transition matrix
  - Posterior interval coverage

6. **Application integration**
- `advanced_analysis.py` updated:
  - Keeps MSD slope baseline (for backward compatibility)
  - Adds parallel advanced inference outputs:
    - `bias_aware_diffusion_inference`
    - `switching_diffusion_hmm`
    - `bayesian_diffusion_posterior` (optional)
  - Captures inference warnings without failing entire analysis

## Metrics to Track (Synthetic and Real Data)

1. **Bias-aware diffusion population inference**
- D bias, variance, and MAE across realistic synthetic conditions
- Fraction recovery error vs out-of-focus severity
- Robustness to localization uncertainty mis-specification

2. **Switching diffusion HMM**
- Transition matrix MAE
- Dwell-time relative error
- State occupancy recovery error
- Posterior state calibration

3. **Bayesian uncertainty**
- 95% interval coverage on synthetic truth
- Effective sample size and acceptance diagnostics
- Sensitivity to priors

4. **Representation learning**
- Accuracy and calibration across imaging domains
- Robustness under domain-randomized perturbations
- Performance drop from synthetic-to-real transfer

## Next Steps

1. Add curated synthetic benchmark scripts and CI regression thresholds.
2. Add optional variational Bayes backend for faster large-scale HMM.
3. Integrate curated labeled trajectories for representation fine-tuning.
4. Add calibration dashboards to report uncertainty quality in UI/export.
