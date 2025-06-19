"""
Universal Alignment Wrapper for All Biophysical Analyses
Provides consistent nuclear alignment preprocessing across all analysis methods
"""

import numpy as np
from typing import Dict, Any, Tuple
import warnings

def wrap_analysis_with_alignment(analysis_func, analysis_name: str):
    """
    Wrapper function that adds nuclear alignment to any biophysical analysis
    
    Args:
        analysis_func: The original analysis function
        analysis_name: Name of the analysis for logging
    
    Returns:
        Wrapped function with alignment preprocessing
    """
    
    def aligned_analysis(self, data_info: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis with nuclear alignment preprocessing
        """
        
        original_image_data = data_info.get('image_data')
        
        if original_image_data is None:
            return {'status': 'error', 'message': 'No image data provided'}
        
        # Check if alignment is enabled
        enable_alignment = parameters.get('nuclear_alignment', True)
        
        if enable_alignment and original_image_data.ndim >= 3:
            try:
                # Import alignment module
                from alignment_integration import apply_nuclear_alignment, create_alignment_summary
                
                # Apply nuclear alignment
                aligned_data, alignment_applied, alignment_quality = apply_nuclear_alignment(
                    original_image_data, parameters, analysis_name
                )
                
                # Update data_info with aligned data
                aligned_data_info = data_info.copy()
                aligned_data_info['image_data'] = aligned_data
                
                # Run original analysis with aligned data
                results = analysis_func(self, aligned_data_info, parameters)
                
                # Add alignment information to results
                if isinstance(results, dict):
                    alignment_summary = create_alignment_summary(alignment_applied, alignment_quality, analysis_name)
                    
                    # Add alignment info at top level
                    results['nuclear_alignment'] = alignment_summary
                    
                    # Add to analysis summary if it exists
                    if 'analysis_summary' in results:
                        results['analysis_summary'].update({
                            'nuclear_alignment_applied': alignment_applied,
                            'data_reliability': alignment_summary.get('reliability_impact', 'Unknown')
                        })
                    
                    # For multi-channel results, add to each channel
                    for key, value in results.items():
                        if isinstance(value, dict) and 'status' in value and key != 'nuclear_alignment':
                            if 'analysis_summary' not in value:
                                value['analysis_summary'] = {}
                            value['analysis_summary'].update({
                                'nuclear_alignment_applied': alignment_applied,
                                'data_reliability': alignment_summary.get('reliability_impact', 'Unknown')
                            })
                
                return results
                
            except ImportError:
                warnings.warn(f"{analysis_name}: Nuclear alignment module not available")
                return analysis_func(self, data_info, parameters)
            except Exception as e:
                warnings.warn(f"{analysis_name}: Alignment failed - {str(e)}, proceeding without alignment")
                return analysis_func(self, data_info, parameters)
        else:
            # Run analysis without alignment
            results = analysis_func(self, data_info, parameters)
            
            # Add alignment status to results
            if isinstance(results, dict):
                no_alignment_summary = {
                    'nuclear_alignment_applied': False,
                    'alignment_status': 'Disabled' if not enable_alignment else 'Not applicable (2D data)',
                    'recommendation': f'Enable nuclear alignment for {analysis_name} analysis' if original_image_data.ndim >= 3 else 'Not applicable for 2D data',
                    'reliability_impact': 'Moderate - motion artifacts may affect results' if original_image_data.ndim >= 3 else 'N/A'
                }
                
                results['nuclear_alignment'] = no_alignment_summary
                
                # Add to analysis summary if it exists
                if 'analysis_summary' in results:
                    results['analysis_summary'].update({
                        'nuclear_alignment_applied': False,
                        'data_reliability': no_alignment_summary['reliability_impact']
                    })
                
                # For multi-channel results, add to each channel
                for key, value in results.items():
                    if isinstance(value, dict) and 'status' in value and key != 'nuclear_alignment':
                        if 'analysis_summary' not in value:
                            value['analysis_summary'] = {}
                        value['analysis_summary'].update({
                            'nuclear_alignment_applied': False,
                            'data_reliability': no_alignment_summary['reliability_impact']
                        })
            
            return results
    
    return aligned_analysis

def get_alignment_enabled_parameters(analysis_type: str) -> Dict[str, Any]:
    """
    Get default parameters with nuclear alignment enabled for specific analysis
    
    Args:
        analysis_type: Type of analysis (RICS, FCS, iMSD, etc.)
    
    Returns:
        Default parameters with alignment settings
    """
    
    from alignment_integration import get_alignment_parameters
    
    # Get analysis-specific alignment parameters
    alignment_params = get_alignment_parameters(analysis_type)
    
    # Standard parameters that apply to all analyses
    base_params = {
        'nuclear_alignment': True,
        'alignment_quality_threshold': 0.6,  # Minimum quality score for reliable results
        'alignment_warning_threshold': 0.4,  # Quality below this triggers warnings
    }
    
    # Merge with analysis-specific parameters
    base_params.update(alignment_params)
    
    return base_params

def validate_analysis_reliability(results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
    """
    Validate the reliability of analysis results based on alignment quality
    
    Args:
        results: Analysis results dictionary
        analysis_type: Type of analysis performed
    
    Returns:
        Validation report with recommendations
    """
    
    if 'nuclear_alignment' not in results:
        return {
            'reliability_assessment': 'Unknown',
            'recommendation': 'Results lack alignment information',
            'confidence_level': 'Low'
        }
    
    alignment_info = results['nuclear_alignment']
    alignment_applied = alignment_info.get('nuclear_alignment_applied', False)
    
    if not alignment_applied:
        return {
            'reliability_assessment': 'Moderate',
            'recommendation': f'Enable nuclear alignment for {analysis_type} to improve result reliability',
            'confidence_level': 'Medium',
            'potential_issues': ['Motion artifacts may affect quantitative measurements',
                               'Temporal drift could bias correlation analyses',
                               'Results should be interpreted with caution']
        }
    
    reliability_score = alignment_info.get('reliability_score', 0)
    quality_assessment = alignment_info.get('quality_assessment', 'Unknown')
    
    if reliability_score >= 0.8:
        confidence_level = 'High'
        reliability_assessment = 'Excellent'
        recommendation = f'{analysis_type} results are highly reliable for quantitative analysis'
        potential_issues = []
    elif reliability_score >= 0.6:
        confidence_level = 'Medium-High'
        reliability_assessment = 'Good'
        recommendation = f'{analysis_type} results are suitable for most scientific applications'
        potential_issues = ['Minor motion artifacts may be present',
                          'Consider validation with independent measurements']
    elif reliability_score >= 0.4:
        confidence_level = 'Medium'
        reliability_assessment = 'Moderate'
        recommendation = f'{analysis_type} results should be interpreted with caution'
        potential_issues = ['Significant motion artifacts detected',
                          'Results may have reduced quantitative accuracy',
                          'Consider alternative alignment methods or manual correction']
    else:
        confidence_level = 'Low'
        reliability_assessment = 'Poor'
        recommendation = f'{analysis_type} results have low reliability due to poor alignment'
        potential_issues = ['Severe motion artifacts affect measurements',
                          'Quantitative results may be unreliable',
                          'Manual inspection and correction recommended',
                          'Consider alternative analysis approaches']
    
    return {
        'reliability_assessment': reliability_assessment,
        'recommendation': recommendation,
        'confidence_level': confidence_level,
        'alignment_quality': quality_assessment,
        'reliability_score': reliability_score,
        'potential_issues': potential_issues
    }

def create_comprehensive_analysis_report(results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
    """
    Create comprehensive analysis report including alignment and reliability assessment
    
    Args:
        results: Analysis results dictionary
        analysis_type: Type of analysis performed
    
    Returns:
        Comprehensive report with all assessments
    """
    
    reliability_validation = validate_analysis_reliability(results, analysis_type)
    
    # Extract key metrics from results
    analysis_summary = results.get('analysis_summary', {})
    alignment_info = results.get('nuclear_alignment', {})
    
    # Count successful analysis channels/regions
    successful_analyses = 0
    total_analyses = 0
    
    for key, value in results.items():
        if isinstance(value, dict) and 'status' in value:
            total_analyses += 1
            if value.get('status') == 'success':
                successful_analyses += 1
    
    report = {
        'analysis_type': analysis_type,
        'overall_status': 'success' if successful_analyses > 0 else 'failed',
        'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0,
        'alignment_summary': alignment_info,
        'reliability_assessment': reliability_validation,
        'key_findings': analysis_summary,
        'recommendations': []
    }
    
    # Generate recommendations based on results
    if reliability_validation['confidence_level'] in ['Low', 'Medium']:
        report['recommendations'].extend(reliability_validation.get('potential_issues', []))
    
    if not alignment_info.get('nuclear_alignment_applied', False):
        report['recommendations'].append(f'Enable nuclear alignment for {analysis_type} analysis')
    
    if successful_analyses < total_analyses:
        report['recommendations'].append('Some analysis regions failed - check input data quality')
    
    if len(report['recommendations']) == 0:
        report['recommendations'].append('Analysis completed successfully with high reliability')
    
    return report