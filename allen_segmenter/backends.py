"""
Segmentation Backends for Allen Cell Segmenter Integration
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Literal, List
import numpy as np
import warnings
import importlib
import sys

from .utils import supports_segmenter_ml, to_segmenter_array

class SegmentationBackend(ABC):
    """Abstract base class for segmentation backends."""

    @abstractmethod
    def segment(self, volume: np.ndarray, *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Segment the input volume.

        Args:
            volume: Input 3D numpy array (Z, Y, X).
            structure_id: Identifier for the structure to segment (e.g., 'LAMP1').
            workflow_id: Identifier for the specific workflow/model.
            config: Dictionary of parameter overrides.

        Returns:
            Binary 3D mask (Z, Y, X).
        """
        pass

    @abstractmethod
    def get_available_structures(self) -> List[str]:
        """Return a list of available structure IDs."""
        pass

class ClassicSegmenterBackend(SegmentationBackend):
    """Backend using aicssegmentation classic workflows."""

    def __init__(self):
        try:
            import aicssegmentation
            from aicssegmentation import structure_wrapper
            self.structure_wrapper = structure_wrapper
        except ImportError:
            raise ImportError("aicssegmentation is not installed.")
        
        self._registry = self._build_registry()

    def _build_registry(self) -> Dict[str, Any]:
        """
        Dynamically discover available structure wrappers.
        """
        registry = {}
        # Hardcoded list of known wrappers in aicssegmentation 0.5.x to look for
        # This is safer than pure introspection which might pick up internal modules
        known_structures = [
            "seg_actb", "seg_a6p", "seg_atp2a2", "seg_cetn2", "seg_connexin",
            "seg_desmoplakin", "seg_dna", "seg_fbl", "seg_h2b", "seg_lamp1",
            "seg_lmnb1", "seg_myh10", "seg_npm1", "seg_nup153", "seg_pxn",
            "seg_rab5a", "seg_sec61b", "seg_slc25a17", "seg_smc1a", "seg_son",
            "seg_st6gal1", "seg_tjp1", "seg_tomm20", "seg_tnnc1", "seg_tub"
        ]

        for name in known_structures:
            try:
                module_name = f"aicssegmentation.structure_wrapper.{name}"
                module = importlib.import_module(module_name)
                # The wrapper function is usually named 'Wrapper' class or a function
                # In 0.5.2, it's often a class inheriting from StructureWrapper
                # or a function `Workflow_...`
                # Let's look for a 'Wrapper' class
                if hasattr(module, 'Wrapper'):
                    # structure_id is usually the part after 'seg_'
                    struct_id = name.replace("seg_", "").upper()
                    registry[struct_id] = module.Wrapper
            except ImportError:
                continue
        
        return registry

    def get_available_structures(self) -> List[str]:
        return list(self._registry.keys())

    def segment(self, volume: np.ndarray, *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        if structure_id is None:
            raise ValueError("structure_id is required for ClassicSegmenterBackend")
        
        structure_id = structure_id.upper()
        if structure_id not in self._registry:
            raise ValueError(f"Unknown structure_id: {structure_id}. Available: {list(self._registry.keys())}")
        
        # Convert input
        # Classic workflows expect (Z, Y, X) float normalized 0-1 usually, but the wrapper handles some pre-processing
        # We'll ensure it's a numpy array
        img = to_segmenter_array(volume)
        
        # Instantiate wrapper
        WrapperClass = self._registry[structure_id]
        
        # The Wrapper constructor might take parameters. 
        # Usually: Wrapper(workflow_config)
        # If config is provided, we pass it.
        
        # Note: The API for structure wrappers in aicssegmentation can vary.
        # Typically: wrapper = Wrapper(); result = wrapper.segment(image)
        
        try:
            wrapper = WrapperClass()
            
            # Apply parameter overrides if config is present
            # This depends on how the wrapper exposes parameters.
            # Many wrappers have a 'workflow_config' attribute.
            if config:
                # This is a simplification. Real integration might need deep merging of config.
                # For now, we assume the user knows the config structure.
                if hasattr(wrapper, 'workflow_config'):
                    wrapper.workflow_config.update(config)
            
            # Run segmentation
            # The segment method usually returns the result
            result = wrapper.segment(img)
            
            # Result might be a simple array or a dictionary/object depending on the wrapper
            if isinstance(result, np.ndarray):
                return result > 0 # Ensure binary
            elif isinstance(result, list) and len(result) > 0:
                 # Some might return steps
                 return result[-1] > 0
            else:
                # Fallback
                return np.array(result) > 0

        except Exception as e:
            raise RuntimeError(f"Segmentation failed for {structure_id}: {e}")

class MlSegmenterBackend(SegmentationBackend):
    """Backend using segmenter_model_zoo ML workflows."""

    def __init__(self):
        if not supports_segmenter_ml():
            raise RuntimeError("ML Segmentation is not supported in this environment.")
        
        try:
            import segmenter_model_zoo
            from segmenter_model_zoo.zoo import ModelZoo
            self.zoo = ModelZoo()
        except ImportError:
            raise ImportError("segmenter_model_zoo is not installed.")

    def get_available_structures(self) -> List[str]:
        # Map model names to structures
        models = self.zoo.get_model_names()
        return models

    def segment(self, volume: np.ndarray, *, 
                structure_id: Optional[str] = None, 
                workflow_id: Optional[str] = None, 
                config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        
        # In ML backend, structure_id or workflow_id can map to the model name
        model_name = workflow_id or structure_id
        
        if not model_name:
            raise ValueError("workflow_id or structure_id (model name) is required for MlSegmenterBackend")
            
        if model_name not in self.zoo.get_model_names():
             raise ValueError(f"Unknown model: {model_name}. Available: {self.zoo.get_model_names()}")

        # Get the model
        model = self.zoo.get_model(model_name)
        
        # Prepare input
        img = to_segmenter_array(volume)
        
        # Run inference
        # The API for model.predict might vary, but typically accepts the image
        try:
            # Check if we need to move to GPU
            # The model zoo usually handles this if configured, but we can force it
            
            result = model.predict(img)
            
            # Result is usually a probability map or binary mask
            # If probability, threshold it
            if result.dtype == float or np.issubdtype(result.dtype, np.floating):
                return result > 0.5
            return result > 0
            
        except Exception as e:
            raise RuntimeError(f"ML Segmentation failed for {model_name}: {e}")

def get_segmenter_backend(mode: Literal["classic", "ml", "auto"] = "auto") -> SegmentationBackend:
    """Factory function to get the appropriate segmentation backend."""
    
    if mode == "ml":
        return MlSegmenterBackend()
    elif mode == "classic":
        return ClassicSegmenterBackend()
    else: # auto
        if supports_segmenter_ml():
            try:
                return MlSegmenterBackend()
            except Exception:
                warnings.warn("ML backend failed to initialize, falling back to classic.")
                return ClassicSegmenterBackend()
        else:
            return ClassicSegmenterBackend()
