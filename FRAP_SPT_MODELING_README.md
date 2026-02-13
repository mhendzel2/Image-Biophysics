# FRAP/SPT Mechanistic Modeling System

## Overview

This module provides mechanistically interpretable FRAP (Fluorescence Recovery After Photobleaching) and SPT (Single Particle Tracking) analysis for biophysical research.

**Design Philosophy**: Treat this as a biophysics methods paper encoded as software. Every modeling choice must be defensible in grant applications and manuscript reviews.

## Key Features

### ✅ What This System Provides

- **Mechanistic models** based on reaction-diffusion PDEs, not empirical exponentials
- **Explicit mass conservation** tracking and enforcement
- **Likelihood-based fitting** with proper statistical inference
- **Parameter identifiability** testing and reporting
- **Joint FRAP-SPT analysis** to resolve parameter degeneracies
- **Model selection** tools (AIC, BIC, evidence ratios)

### ❌ What This System Rejects

- Black-box curve fitting
- Purely empirical exponentials without physical justification
- Diffusion coefficients inferred without geometry scaling
- Binding rates inferred without independent validation (SPT)
- Coalescence invoked to "explain" poor fits

## Module Structure

```
frap_models/         # Core FRAP models
├── base.py          # Abstract interface
├── simulators.py    # PDE integration engine
├── reaction_diffusion.py      # Classical RD model
├── mass_conserving.py         # MCRD for condensates
└── coalescence.py   # Optional exchange models (use with caution)

frap_fitting/        # Statistical fitting
├── likelihoods.py   # Gaussian/Poisson likelihoods
├── fit_rd.py        # RD model fitting
├── fit_mcrd.py      # MCRD model fitting
└── model_selection.py         # AIC/BIC comparison

spt_models/          # SPT integration
├── dwell_time.py    # Dwell time analysis
├── state_transition.py        # Hidden Markov Models
└── joint_frap_spt.py          # Joint FRAP-SPT fitting

notebooks/           # Demonstrations
├── FRAP_Model_Demonstration.ipynb
└── FRAP_SPT_Joint_Fit.ipynb

tests/               # Test suite
├── test_mass_conservation.py
├── test_parameter_identifiability.py
└── test_rd_recovery.py
```

## Quick Start

### 1. Basic FRAP Simulation

```python
from frap_models import ReactionDiffusionModel
import numpy as np

# Initialize model
model = ReactionDiffusionModel()

# Define geometry
geometry = {
    'shape': (100, 100),
    'spacing': 0.1,  # μm
    'bleach_region': {
        'type': 'circular',
        'center': (5.0, 5.0),
        'radius': 1.0,
        'bleach_depth': 0.9
    }
}

# Parameters
params = {
    'D': 5.0,      # μm²/s
    'k_on': 1.0,   # 1/s
    'k_off': 1.0,  # 1/s
    'bleach_depth': 0.9
}

# Simulate
timepoints = np.array([0, 1, 5, 10, 20, 40])
recovery = model.simulate(params, geometry, timepoints)
```

### 2. Fit FRAP Data

```python
from frap_fitting import fit_reaction_diffusion

# Fit model to observed data
result = fit_reaction_diffusion(
    observed_recovery=observed_data,
    timepoints=timepoints,
    geometry=geometry,
    likelihood_type='gaussian'
)

print(f"Best-fit D: {result['params']['D']:.2f} μm²/s")
print(f"AIC: {result['aic']:.2f}")
```

### 3. Joint FRAP-SPT Analysis

```python
from spt_models import fit_joint_model

frap_data = {
    'recovery': observed_recovery,
    'timepoints': timepoints,
    'geometry': geometry
}

spt_data = {
    'bound_dwells': bound_dwell_times,
    'unbound_dwells': unbound_dwell_times
}

# Joint fit resolves parameter degeneracies
result = fit_joint_model(frap_data, spt_data)
```

## Model Descriptions

### Reaction-Diffusion (RD)

**Equations:**
```
∂F/∂t = D ∇²F - k_on·F + k_off·B
∂B/∂t = k_on·F - k_off·B
```

**Parameters:**
- `D`: Diffusion coefficient of free species [μm²/s]
- `k_on`: Binding rate [1/s]
- `k_off`: Unbinding rate [1/s]

**Use when:**
- Molecules freely diffuse and bind to immobile structures
- No compartmentalization
- Exchange with large reservoir is valid

### Mass-Conserving RD (MCRD)

**Key constraint:**
```
∫(c_condensed + c_dilute) dV = constant
```

**Parameters:**
- `D_dilute`: Diffusion in dilute phase [μm²/s]
- `D_condensed`: Diffusion in condensed phase [μm²/s]
- `k_in`: Condensation rate [1/s]
- `k_out`: Dissolution rate [1/s]

**Use when:**
- Analyzing biomolecular condensates
- No exchange with external reservoir
- Need to track phase partitioning

### Coalescence (EXPERIMENTAL)

**⚠️ Warning:** Coalescence terms are weakly constrained by FRAP alone.

**Use only when:**
1. You have complementary data (particle tracking)
2. Coalescence is directly observable
3. Testing specific hypotheses about fusion

**DO NOT use** to "explain away" poor fits.

## Testing

Run the test suite to validate installation:

```bash
python -m unittest discover tests -v
```

Key tests:
- **Mass conservation**: Verifies total fluorescence is conserved post-bleach
- **Parameter identifiability**: Tests for degeneracies in fits
- **Recovery behavior**: Validates monotonic recovery and geometry scaling

## Demonstration Notebooks

### FRAP_Model_Demonstration.ipynb

Demonstrates:
- Model equations and implementation
- Parameter sensitivity analysis
- Geometry scaling tests
- Recovery regime identification

### FRAP_SPT_Joint_Fit.ipynb

Shows:
- FRAP-only fitting (demonstrates degeneracy)
- SPT-only analysis (constrains kinetics)
- Joint fitting (resolves ambiguity)
- Parameter comparison and validation

## Best Practices

### For Grant Applications

1. **Always test identifiability** before claiming parameter values
2. **Report confidence intervals** from bootstrap or likelihood profiles
3. **Use model selection** (AIC/BIC) to justify model choice
4. **Combine with SPT** when possible to constrain kinetics

### For Manuscripts

1. **Show mass conservation** explicitly in supplementary figures
2. **Report Courant conditions** for numerical stability
3. **Demonstrate geometry scaling** to validate diffusion interpretation
4. **Compare to analytic limits** where available

### Common Pitfalls to Avoid

❌ Fitting diffusion coefficient without varying bleach geometry  
❌ Claiming k_on, k_off from FRAP alone without SPT validation  
❌ Ignoring parameter uncertainty  
❌ Using coalescence terms without independent evidence  
❌ Presenting empirical exponentials as mechanistic models  

## Dependencies

- `numpy >= 1.23.0`
- `scipy >= 1.11.0`
- `matplotlib` (for notebooks)

## References

### Theoretical Background

- **Classical FRAP**: Axelrod et al. (1976) Biophys J.
- **Reaction-diffusion**: Sprague et al. (2004) Biophys J.
- **Mass conservation**: Beaudouin et al. (2006) Nature.
- **Parameter identifiability**: Mueller et al. (2013) Methods.

### Model Selection

- **AIC/BIC**: Burnham & Anderson (2002) Model Selection.
- **Evidence ratios**: Kass & Raftery (1995) JASA.

## Support and Contributing

For questions about model selection or parameter interpretation, please consult:
1. The demonstration notebooks
2. Test cases for examples
3. Inline documentation in model classes

When reporting issues, please include:
- Model type and parameters
- Geometry specification
- Observed vs expected behavior
- Test case that reproduces the issue

## License

See main repository LICENSE file.

## Citation

If you use this FRAP/SPT modeling system, please cite:

- The Image-Biophysics repository
- Relevant theoretical papers for models used
- This implementation (DOI: pending)

## Acknowledgments

This implementation follows best practices from the biophysical modeling community and incorporates feedback from reviewers of CIHR/NIH grant applications.
