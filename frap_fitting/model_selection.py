"""
Model Selection for FRAP Analysis

Compare different FRAP models using information criteria and evidence ratios.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import warnings

from .likelihoods import compute_aic, compute_bic, compute_aicc, compute_evidence_ratio


class ModelSelection:
    """
    Model selection and comparison for FRAP data.
    
    Computes AIC, BIC, and evidence ratios to compare models.
    """
    
    def __init__(self):
        """Initialize model selection."""
        self.models = {}
        self.results = {}
    
    def add_model(
        self,
        name: str,
        log_likelihood: float,
        n_params: int,
        n_data: int,
        predicted: np.ndarray = None,
        params: Dict[str, float] = None
    ):
        """
        Add a model to comparison.
        
        Parameters
        ----------
        name : str
            Model name
        log_likelihood : float
            Maximum log-likelihood
        n_params : int
            Number of fitted parameters
        n_data : int
            Number of data points
        predicted : np.ndarray, optional
            Model predictions
        params : dict, optional
            Fitted parameters
        """
        aic = compute_aic(log_likelihood, n_params)
        bic = compute_bic(log_likelihood, n_params, n_data)
        aicc = compute_aicc(log_likelihood, n_params, n_data)
        
        self.results[name] = {
            'log_likelihood': log_likelihood,
            'n_params': n_params,
            'n_data': n_data,
            'aic': aic,
            'bic': bic,
            'aicc': aicc,
            'predicted': predicted,
            'params': params
        }
    
    def get_best_model(self, criterion: str = 'aic') -> str:
        """
        Get best model according to criterion.
        
        Parameters
        ----------
        criterion : str
            'aic', 'bic', or 'aicc'
        
        Returns
        -------
        best_model : str
            Name of best model
        """
        if not self.results:
            raise ValueError("No models added yet")
        
        values = {name: res[criterion] for name, res in self.results.items()}
        best_model = min(values, key=values.get)
        
        return best_model
    
    def compute_weights(self, criterion: str = 'aic') -> Dict[str, float]:
        """
        Compute Akaike weights for model averaging.
        
        Weight_i = exp(-Δ_i/2) / Σ exp(-Δ_j/2)
        where Δ_i = AIC_i - AIC_min
        
        Parameters
        ----------
        criterion : str
            Information criterion to use
        
        Returns
        -------
        weights : dict
            Model weights (sum to 1)
        """
        values = {name: res[criterion] for name, res in self.results.items()}
        min_value = min(values.values())
        
        # Compute relative likelihoods
        rel_likes = {
            name: np.exp(-(val - min_value) / 2)
            for name, val in values.items()
        }
        
        # Normalize to weights
        total = sum(rel_likes.values())
        weights = {name: like / total for name, like in rel_likes.items()}
        
        return weights
    
    def compare_models(
        self,
        model1: str,
        model2: str,
        criterion: str = 'aic'
    ) -> Dict[str, Any]:
        """
        Compare two models directly.
        
        Parameters
        ----------
        model1 : str
            First model name
        model2 : str
            Second model name
        criterion : str
            Criterion for comparison
        
        Returns
        -------
        comparison : dict
            Comparison results
        """
        if model1 not in self.results or model2 not in self.results:
            raise ValueError("Both models must be added first")
        
        val1 = self.results[model1][criterion]
        val2 = self.results[model2][criterion]
        
        delta = abs(val1 - val2)
        evidence_ratio = compute_evidence_ratio(min(val1, val2), max(val1, val2))
        
        if val1 < val2:
            better = model1
            worse = model2
        else:
            better = model2
            worse = model1
        
        # Interpret evidence ratio
        if evidence_ratio > 10:
            interpretation = "Strong evidence"
        elif evidence_ratio > 3:
            interpretation = "Moderate evidence"
        elif evidence_ratio > 1.5:
            interpretation = "Weak evidence"
        else:
            interpretation = "Negligible difference"
        
        return {
            'better_model': better,
            'worse_model': worse,
            f'{criterion}_difference': delta,
            'evidence_ratio': evidence_ratio,
            'interpretation': interpretation
        }
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate summary of all models.
        
        Returns
        -------
        summary : dict
            Summary statistics
        """
        if not self.results:
            return {}
        
        # Sort by AIC
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['aic']
        )
        
        summary = {
            'n_models': len(self.results),
            'best_model_aic': sorted_models[0][0],
            'best_model_bic': self.get_best_model('bic'),
            'models': []
        }
        
        for name, res in sorted_models:
            summary['models'].append({
                'name': name,
                'log_likelihood': res['log_likelihood'],
                'n_params': res['n_params'],
                'aic': res['aic'],
                'bic': res['bic'],
                'aicc': res['aicc']
            })
        
        return summary
    
    def flag_unidentifiable_models(self, threshold: float = 2.0) -> List[str]:
        """
        Flag models with very similar information criteria.
        
        If multiple models have AIC within threshold, parameters may
        not be identifiable from data alone.
        
        Parameters
        ----------
        threshold : float
            AIC difference threshold
        
        Returns
        -------
        flags : list
            List of warnings about unidentifiable models
        """
        if len(self.results) < 2:
            return []
        
        flags = []
        aics = {name: res['aic'] for name, res in self.results.items()}
        min_aic = min(aics.values())
        
        similar_models = [
            name for name, aic in aics.items()
            if abs(aic - min_aic) < threshold
        ]
        
        if len(similar_models) > 1:
            flags.append(
                f"Warning: Models {similar_models} have similar AIC (Δ < {threshold}). "
                "Parameters may not be uniquely identifiable. "
                "Consider additional experiments (e.g., SPT) to resolve ambiguity."
            )
        
        return flags


def compare_models(
    results_list: List[Dict[str, Any]],
    model_names: List[str] = None
) -> ModelSelection:
    """
    Convenience function to compare multiple model fits.
    
    Parameters
    ----------
    results_list : list
        List of fit results from fit_rd, fit_mcrd, etc.
    model_names : list, optional
        Names for models
    
    Returns
    -------
    selection : ModelSelection
        Model selection object with comparison
    """
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(results_list))]
    
    selection = ModelSelection()
    
    for name, result in zip(model_names, results_list):
        selection.add_model(
            name,
            result['log_likelihood'],
            len(result['params']),
            len(result['predicted']),
            result['predicted'],
            result['params']
        )
    
    return selection
