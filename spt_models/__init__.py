"""
SPT (Single Particle Tracking) Models

Integration of SPT analysis with FRAP:
- Dwell time distributions
- State transition models
- Joint FRAP-SPT fitting
- Bias-aware diffusion population inference
- Switching diffusion HMM with uncertainty
- Bayesian posterior workflows
"""

from .dwell_time import DwellTimeModel, fit_dwell_times
from .state_transition import StateTransitionModel, HiddenMarkovModel
from .joint_frap_spt import JointFRAPSPT, fit_joint_model
from .spot_on import SpotOnConfig, SpotOnLikeInference
from .switching_diffusion import SwitchingDiffusionHMM, SwitchingHMMConfig
from .bayesian import BayesianDiffusionInference, BAYES_TRAJ_AVAILABLE
from .trajectory_utils import (
    DisplacementDataset,
    extract_displacements,
    extract_step_sequences,
    normalize_tracks,
    scale_track_coordinates,
)
from .trajectory_representation import (
    SyntheticTrajectoryConfig,
    TrajectoryFeatureEmbedder,
    TrajectoryTransformerEncoder,
    generate_synthetic_dataset,
    simulate_trajectory,
)
from .benchmarking import (
    TwoStateSimulationConfig,
    simulate_two_state_tracks,
    diffusion_recovery_metrics,
    fraction_recovery_metrics,
    transition_recovery_metrics,
    dwell_time_from_transition,
    posterior_calibration_metrics,
)

__all__ = [
    'DwellTimeModel',
    'fit_dwell_times',
    'StateTransitionModel',
    'HiddenMarkovModel',
    'JointFRAPSPT',
    'fit_joint_model',
    'SpotOnConfig',
    'SpotOnLikeInference',
    'SwitchingDiffusionHMM',
    'SwitchingHMMConfig',
    'BayesianDiffusionInference',
    'BAYES_TRAJ_AVAILABLE',
    'DisplacementDataset',
    'normalize_tracks',
    'scale_track_coordinates',
    'extract_displacements',
    'extract_step_sequences',
    'SyntheticTrajectoryConfig',
    'simulate_trajectory',
    'generate_synthetic_dataset',
    'TrajectoryFeatureEmbedder',
    'TrajectoryTransformerEncoder',
    'TwoStateSimulationConfig',
    'simulate_two_state_tracks',
    'diffusion_recovery_metrics',
    'fraction_recovery_metrics',
    'transition_recovery_metrics',
    'dwell_time_from_transition',
    'posterior_calibration_metrics',
]
