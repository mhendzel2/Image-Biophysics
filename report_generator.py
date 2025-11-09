"""
Automated Report Generation Module
Creates comprehensive analysis reports based on imported data and results
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import io
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class AutomatedReportGenerator:
    """Generates comprehensive analysis reports based on data and results"""

    def __init__(self):
        self.report_templates = {
            'microscopy_analysis': self._generate_microscopy_report,
            'fcs_analysis': self._generate_fcs_report,
            'specialized_physics': self._generate_physics_report,
            'ai_enhancement': self._generate_ai_report,
            'comprehensive': self._generate_comprehensive_report
        }

    def generate_report(self, data_info: Dict[str, Any], analysis_results: Dict[str, Any],
                       specialized_results: Dict[str, Any], enhanced_data: Any = None,
                       report_type: str = 'comprehensive') -> Dict[str, Any]:
        """Generate automated report based on data and results"""

        report_data = {
            'metadata': self._extract_metadata(data_info),
            'data_summary': self._analyze_data_characteristics(data_info),
            'analysis_summary': self._summarize_analysis_results(analysis_results),
            'specialized_summary': self._summarize_specialized_results(specialized_results),
            'recommendations': self._generate_recommendations(data_info, analysis_results, specialized_results),
            'timestamp': datetime.now().isoformat()
        }

        # Generate report content based on type
        if report_type in self.report_templates:
            report_content = self.report_templates[report_type](report_data, data_info, analysis_results, specialized_results)
        else:
            report_content = self._generate_comprehensive_report(report_data, data_info, analysis_results, specialized_results)

        return {
            'report_data': report_data,
            'report_content': report_content,
            'report_type': report_type,
            'generation_time': datetime.now().isoformat()
        }

    def _extract_metadata(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize metadata from data"""
        metadata = {
            'filename': data_info.get('filename', 'Unknown'),
            'format': data_info.get('format', 'Unknown'),
            'data_type': data_info.get('data_type', 'Unknown'),
            'dimensions': str(data_info.get('shape', 'Unknown')),
            'pixel_size': data_info.get('pixel_size', 'Unknown'),
            'time_points': data_info.get('time_points', 'Unknown'),
            'channels': data_info.get('channels', 'Unknown'),
            'channel_names': data_info.get('channel_names', []),
            'acquisition_date': data_info.get('acquisition_date', 'Unknown'),
            'microscope_type': self._infer_microscope_type(data_info.get('format', '')),
            'file_size': data_info.get('file_size', 'Unknown')
        }

        # Add computed metrics
        if data_info.get('shape'):
            shape = data_info['shape']
            if len(shape) >= 2:
                metadata['total_pixels'] = shape[-1] * shape[-2]
                if len(shape) >= 3:
                    metadata['total_frames'] = shape[0]
                    metadata['dataset_size'] = f"{shape[0]} × {shape[-2]} × {shape[-1]}"

        return metadata

    def _analyze_data_characteristics(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data characteristics and quality metrics"""
        characteristics = {}
        return characteristics

    def _summarize_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize analysis results"""
        summary = {}
        return summary

    def _summarize_specialized_results(self, specialized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize specialized results"""
        summary = {}
        return summary

    def _generate_recommendations(self, data_info: Dict[str, Any], 
                                 analysis_results: Dict[str, Any],
                                 specialized_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        return recommendations

    def _infer_microscope_type(self, format_str: str) -> str:
        """Infer microscope type from format"""
        if 'czi' in format_str.lower():
            return 'Zeiss Confocal'
        elif 'lif' in format_str.lower():
            return 'Leica Confocal'
        elif 'nd2' in format_str.lower():
            return 'Nikon'
        elif 'oib' in format_str.lower() or 'oif' in format_str.lower():
            return 'Olympus'
        return 'Unknown'

    def _generate_microscopy_report(self, report_data: Dict[str, Any], 
                                    data_info: Dict[str, Any], 
                                    analysis_results: Dict[str, Any], 
                                    specialized_results: Dict[str, Any]) -> str:
        """Generate microscopy-focused report"""
        content = ["# Microscopy Analysis Report\n\n"]
        content.append(f"**Generated:** {report_data['timestamp']}\n\n")
        return "".join(content)

    def _generate_fcs_report(self, report_data: Dict[str, Any], 
                            data_info: Dict[str, Any], 
                            analysis_results: Dict[str, Any], 
                            specialized_results: Dict[str, Any]) -> str:
        """Generate FCS-focused report"""
        content = ["# FCS Analysis Report\n\n"]
        content.append(f"**Generated:** {report_data['timestamp']}\n\n")
        return "".join(content)

    def _generate_physics_report(self, report_data: Dict[str, Any], 
                                data_info: Dict[str, Any], 
                                analysis_results: Dict[str, Any], 
                                specialized_results: Dict[str, Any]) -> str:
        """Generate physics-focused report"""
        content = ["# Specialized Physics Analysis Report\n\n"]
        content.append(f"**Generated:** {report_data['timestamp']}\n\n")
        return "".join(content)

    def _generate_ai_report(self, report_data: Dict[str, Any], 
                           data_info: Dict[str, Any], 
                           analysis_results: Dict[str, Any], 
                           specialized_results: Dict[str, Any]) -> str:
        """Generate AI enhancement report"""
        content = ["# AI Enhancement Report\n\n"]
        content.append(f"**Generated:** {report_data['timestamp']}\n\n")
        return "".join(content)

    def _generate_comprehensive_report(self, report_data: Dict[str, Any], 
                                     data_info: Dict[str, Any], 
                                     analysis_results: Dict[str, Any], 
                                     specialized_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report with more detail."""
        content = []
        
        # Header and metadata
        content.append("# Comprehensive Analysis Report\n\n")
        content.append(f"**Generated:** {report_data['timestamp']}\n\n")
        
        # Metadata section
        if 'metadata' in report_data:
            content.append("## File Metadata\n\n")
            metadata = report_data['metadata']
            for key, value in metadata.items():
                content.append(f"- **{key}**: {value}\n")
            content.append("\n")

        # Segmented FCS section if present
        if 'Segmented FCS' in analysis_results:
            seg_result = analysis_results['Segmented FCS']
            if isinstance(seg_result, dict) and seg_result.get('status') == 'success':
                segments_df = seg_result.get('segments')
                content.append("## Segmented FCS\n")
                content.append(f"Computed {seg_result.get('n_segments', 0)} segments. ")
                content.append(f"Median D: {seg_result.get('median_D_um2_s', 0):.4g} µm²/s | ")
                content.append(f"Median τD: {seg_result.get('median_tauD_s', 0):.4g} s | ")
                content.append(f"Median N: {seg_result.get('median_N_est', 0):.3g}\n\n")
                try:
                    # Include compact CSV (first few lines) for readability
                    if hasattr(segments_df, 'head'):
                        csv_preview = segments_df.head(10).to_csv(index=False)
                        content.append("Segment Summary (first 10):\n")
                        content.append("```\n" + csv_preview + "```\n")
                except Exception:
                    pass
        
        # Analysis results section
        if analysis_results:
            content.append("## Analysis Results\n\n")
            for method, result in analysis_results.items():
                if method != 'Segmented FCS':  # Already handled above
                    content.append(f"### {method}\n")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            content.append(f"- **{key}**: {value}\n")
                    else:
                        content.append(f"{result}\n")
                    content.append("\n")
        
        # Specialized results section
        if specialized_results:
            content.append("## Specialized Analysis\n\n")
            for key, value in specialized_results.items():
                content.append(f"### {key}\n")
                content.append(f"{value}\n\n")
        
        # Recommendations
        if 'recommendations' in report_data and report_data['recommendations']:
            content.append("## Recommendations\n\n")
            for rec in report_data['recommendations']:
                content.append(f"- {rec}\n")
            content.append("\n")
        
        return "".join(content)
