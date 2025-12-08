"""
Batch Processing Module

This module provides the framework for running a defined image processing and analysis 
pipeline on a collection of images from an input directory and saving the results to an 
output directory.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from data_loader import DataLoader
from ai_enhancement import AIEnhancementManager
from analysis import AnalysisManager
from tifffile import imwrite

class BatchProcessor:
    """Manages the execution of a batch processing pipeline."""

    def __init__(self, 
                 enhancement_method: str, 
                 enhancement_params: Dict[str, Any],
                 analysis_tasks: List[str],
                 analysis_params: Dict[str, Any]):
        """
        Initializes the BatchProcessor with the processing pipeline.

        Args:
            enhancement_method: The AI enhancement method to apply.
            enhancement_params: Parameters for the enhancement method.
            analysis_tasks: List of analysis tasks to perform (e.g., ['morphometrics']).
            analysis_params: Parameters for the analysis tasks.
        """
        self.data_loader = DataLoader()
        self.ai_enhancer = AIEnhancementManager()
        self.analyzer = AnalysisManager()

        self.enhancement_method = enhancement_method
        self.enhancement_params = enhancement_params
        self.analysis_tasks = analysis_tasks
        self.analysis_params = analysis_params

    def run(self, input_dir: str, output_dir: str, callback=None):
        """
        Runs the batch processing job.

        Args:
            input_dir: The directory containing images to process.
            output_dir: The directory where results will be saved.
            callback: A function to call with progress updates.
        """
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.czi', '.lif', '.nd2'))]
        total_files = len(image_files)
        all_morphometry_results = []

        for i, filename in enumerate(image_files):
            if callback:
                callback(i, total_files, f"Processing {filename}...")
            
            try:
                # 1. Load Image
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'rb') as f:
                    load_result = self.data_loader.load_image(f)
                if load_result['status'] != 'success':
                    continue
                image = load_result['image_data']

                # 2. Run AI Enhancement / Segmentation
                enh_result = self.ai_enhancer.enhance_image(
                    image, self.enhancement_method, self.enhancement_params
                )
                if enh_result.get('status') != 'success':
                    continue

                segmentation_mask = enh_result.get('segmentation_masks')
                if segmentation_mask is not None:
                    self._save_image(segmentation_mask, f"{Path(filename).stem}_mask.tif", output_dir)

                # 3. Run Analysis
                if 'morphometrics' in self.analysis_tasks and segmentation_mask is not None:
                    voxel_size = self.analysis_params.get('voxel_size', (1.0, 1.0, 1.0))
                    morph_result = self.analyzer.calculate_morphometrics(
                        segmentation_mask, original_image=image, voxel_size=voxel_size
                    )
                    if morph_result.get('status') == 'success':
                        df = morph_result['results_df']
                        df['source_image'] = filename
                        all_morphometry_results.append(df)

                # ... other analysis tasks like percolation could be added here ...

            except Exception as e:
                if callback:
                    callback(i, total_files, f"Error processing {filename}: {e}")
        
        # 4. Save Consolidated Results
        if all_morphometry_results:
            consolidated_df = pd.concat(all_morphometry_results, ignore_index=True)
            consolidated_df.to_csv(os.path.join(output_dir, "consolidated_morphometrics.csv"), index=False)

        if callback:
            callback(total_files, total_files, "Batch processing complete!")

    def _save_image(self, image: np.ndarray, filename: str, output_dir: str):
        """Saves a numpy array as a TIFF image."""
        os.makedirs(output_dir, exist_ok=True)
        imwrite(os.path.join(output_dir, filename), image)
