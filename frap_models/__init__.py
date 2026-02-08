"""
FRAP Models Module

Mechanistically interpretable FRAP analysis using:
- Reaction-diffusion (RD) models
- Mass-conserving reaction-diffusion (MCRD) variants
- Optional coalescence/exchange terms for condensates

All models explicitly encode mass conservation or justify its violation.
"""

from .base import FRAPModel
from .reaction_diffusion import ReactionDiffusionModel
from .mass_conserving import MassConservingRDModel
from .coalescence import CoalescenceModel
from .simulators import FRAPSimulator

__all__ = [
    'FRAPModel',
    'ReactionDiffusionModel',
    'MassConservingRDModel',
    'CoalescenceModel',
    'FRAPSimulator',
]
