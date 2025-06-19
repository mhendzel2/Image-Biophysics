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
        characteristics = {
            'data_quality': 'Good',  # Default
            'temporal_resolution': 'Unknown',
            'spatial_resolution': 'Unknown',
            'signal_characteristics': {},
            'recommended_analyses': []
        }
        
        # Analyze based on data properties
        if data_info.get('shape'):
            shape = data_info['shape']
            
            # Time series analysis
            if len(shape) >= 3 and shape[0] > 1:
                characteristics['temporal_resolution'] = f"{shape[0]} time points"
                characteristics['recommended_analyses'].extend([
                    'Temporal correlation analysis',
                    'Dynamic process tracking',
                    'STICS/RICS analysis'
                ])
            
            # Spatial resolution analysis
            if len(shape) >= 2:
                pixel_count = shape[-1] * shape[-2]
                if pixel_count > 1000000:  # > 1MP
                    characteristics['spatial_resolution'] = 'High resolution'
                elif pixel_count > 100000:  # > 0.1MP
                    characteristics['spatial_resolution'] = 'Medium resolution'
                else:
                    characteristics['spatial_resolution'] = 'Low resolution'
        
        # Multi-channel analysis
        channels = data_info.get('channels', 1)
        if channels > 1:
            characteristics['multichannel'] = True
            characteristics['recommended_analyses'].extend([
                'Cross-correlation analysis',
                'Colocalization studies',
                'Multi-channel dynamics'
            ])
        
        # Format-specific recommendations
        format_type = data_info.get('format', '').lower()
        if 'lsm' in format_type or 'czi' in format_type:
            characteristics['recommended_analyses'].extend([
                'Confocal-specific analysis',
                'Z-stack processing',
                'Point spread function analysis'
            ])
        
        return characteristics
    
    def _summarize_analysis_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize standard analysis results"""
        if not analysis_results:
            return {'status': 'No standard analysis performed'}
        
        summary = {
            'analyses_performed': [],
            'key_findings': {},
            'statistical_summary': {}
        }
        
        # Extract analysis types and results
        for key, result in analysis_results.items():
            if isinstance(result, dict):
                analysis_type = result.get('analysis_type', key)
                summary['analyses_performed'].append(analysis_type)
                
                # Extract key metrics based on analysis type
                if 'fcs' in analysis_type.lower():
                    if 'diffusion_coefficient' in result:
                        summary['key_findings']['diffusion_coefficient'] = f"{result['diffusion_coefficient']:.6f} μm²/s"
                    if 'correlation_time' in result:
                        summary['key_findings']['correlation_time'] = f"{result['correlation_time']:.3f} ms"
                
                elif 'msd' in analysis_type.lower():
                    if 'diffusion_coefficient' in result:
                        summary['key_findings']['msd_diffusion'] = f"{result['diffusion_coefficient']:.6f} μm²/s"
                    if 'alpha_exponent' in result:
                        summary['key_findings']['anomalous_exponent'] = f"{result['alpha_exponent']:.3f}"
                
                elif 'particle' in analysis_type.lower():
                    if 'total_particles' in result:
                        summary['key_findings']['particle_count'] = result['total_particles']
                    if 'average_intensity' in result:
                        summary['key_findings']['avg_particle_intensity'] = f"{result['average_intensity']:.1f}"
        
        return summary
    
    def _summarize_specialized_results(self, specialized_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize specialized physics analysis results"""
        if not specialized_results:
            return {'status': 'No specialized analysis performed'}
        
        summary = {
            'specialized_methods': [],
            'physics_findings': {},
            'advanced_metrics': {}
        }
        
        # Optical flow results
        if 'optical_flow' in specialized_results:
            flow_result = specialized_results['optical_flow']
            summary['specialized_methods'].append(flow_result.get('method', 'Optical Flow'))
            
            if 'avg_displacement' in flow_result:
                summary['physics_findings']['average_displacement'] = f"{flow_result['avg_displacement']:.3f} pixels"
            if 'max_displacement' in flow_result:
                summary['physics_findings']['maximum_displacement'] = f"{flow_result['max_displacement']:.3f} pixels"
            
            # Strain analysis
            if 'strain_results' in flow_result and flow_result['strain_results']:
                strain_data = flow_result['strain_results']
                if 'avg_von_mises_strain' in strain_data:
                    summary['advanced_metrics']['von_mises_strain'] = f"{strain_data['avg_von_mises_strain']:.6f}"
        
        # ICS results
        if 'ics' in specialized_results:
            ics_result = specialized_results['ics']
            summary['specialized_methods'].append(ics_result.get('method', 'Image Correlation Spectroscopy'))
            
            if 'diffusion_coefficient' in ics_result:
                summary['physics_findings']['ics_diffusion'] = f"{ics_result['diffusion_coefficient']:.6f} μm²/s"
            if 'flow_magnitude' in ics_result:
                summary['physics_findings']['flow_velocity'] = f"{ics_result['flow_magnitude']:.3f} μm/s"
            if 'fit_quality' in ics_result:
                summary['advanced_metrics']['correlation_fit_quality'] = f"{ics_result['fit_quality']:.3f}"
        
        return summary
    
    def _generate_recommendations(self, data_info: Dict[str, Any], 
                                analysis_results: Dict[str, Any], 
                                specialized_results: Dict[str, Any]) -> List[str]:
        """Generate analysis recommendations based on data and results"""
        recommendations = []
        
        # Data-based recommendations
        if data_info.get('channels', 1) > 1:
            recommendations.append("Consider cross-correlation analysis between channels for molecular interaction studies")
        
        if data_info.get('shape') and len(data_info['shape']) >= 3:
            if data_info['shape'][0] > 50:  # Many time points
                recommendations.append("Time series data suitable for advanced temporal correlation methods (STICS, temporal ICS)")
        
        # Analysis-based recommendations
        if analysis_results:
            # Check for anomalous diffusion
            for result in analysis_results.values():
                if isinstance(result, dict) and 'alpha_exponent' in result:
                    alpha = result['alpha_exponent']
                    if alpha < 0.8:
                        recommendations.append("Subdiffusive behavior detected - consider confined diffusion models")
                    elif alpha > 1.2:
                        recommendations.append("Superdiffusive behavior detected - investigate active transport mechanisms")
        
        # Specialized analysis recommendations
        if specialized_results:
            if 'optical_flow' in specialized_results:
                flow_result = specialized_results['optical_flow']
                if flow_result.get('avg_displacement', 0) > 1.0:
                    recommendations.append("Significant motion detected - suitable for elastography and force propagation analysis")
            
            if 'ics' in specialized_results:
                ics_result = specialized_results['ics']
                if ics_result.get('flow_magnitude', 0) > 0.1:
                    recommendations.append("Directed flow detected - consider investigating cellular transport mechanisms")
        
        # General recommendations
        if not specialized_results:
            recommendations.append("Consider applying specialized physics methods (optical flow, ICS) for advanced biophysical insights")
        
        if not analysis_results:
            recommendations.append("Apply standard biophysical analysis methods based on your experimental objectives")
        
        return recommendations
    
    def _infer_microscope_type(self, format_str: str) -> str:
        """Infer microscope type from file format"""
        format_lower = format_str.lower()
        
        if 'lsm' in format_lower:
            return 'Zeiss LSM Confocal'
        elif 'czi' in format_lower:
            return 'Zeiss Microscope'
        elif 'lif' in format_lower:
            return 'Leica Microscope'
        elif 'oif' in format_lower:
            return 'Olympus Microscope'
        elif 'tif' in format_lower or 'tiff' in format_lower:
            return 'TIFF-based System'
        elif 'stk' in format_lower:
            return 'MetaMorph System'
        else:
            return 'Unknown System'
    
    def _generate_microscopy_report(self, report_data: Dict[str, Any], 
                                  data_info: Dict[str, Any], 
                                  analysis_results: Dict[str, Any], 
                                  specialized_results: Dict[str, Any]) -> str:
        """Generate microscopy-focused report"""
        content = []
        
        content.append("# Microscopy Data Analysis Report\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Information
        content.append("## Dataset Information\n")
        metadata = report_data['metadata']
        content.append(f"- **Filename:** {metadata['filename']}\n")
        content.append(f"- **Microscope System:** {metadata['microscope_type']}\n")
        content.append(f"- **Format:** {metadata['format']}\n")
        content.append(f"- **Dimensions:** {metadata['dimensions']}\n")
        content.append(f"- **Pixel Size:** {metadata['pixel_size']} μm\n")
        content.append(f"- **Channels:** {metadata['channels']}\n")
        if metadata['channel_names']:
            content.append(f"- **Channel Names:** {', '.join(metadata['channel_names'])}\n")
        content.append("\n")
        
        # Data Characteristics
        content.append("## Data Characteristics\n")
        characteristics = report_data['data_summary']
        content.append(f"- **Spatial Resolution:** {characteristics['spatial_resolution']}\n")
        content.append(f"- **Temporal Resolution:** {characteristics['temporal_resolution']}\n")
        content.append(f"- **Data Quality:** {characteristics['data_quality']}\n")
        content.append("\n")
        
        # Analysis Summary
        if analysis_results:
            content.append("## Analysis Results Summary\n")
            analysis_summary = report_data['analysis_summary']
            content.append(f"- **Methods Applied:** {', '.join(analysis_summary['analyses_performed'])}\n")
            
            if analysis_summary['key_findings']:
                content.append("- **Key Findings:**\n")
                for key, value in analysis_summary['key_findings'].items():
                    content.append(f"  - {key.replace('_', ' ').title()}: {value}\n")
            content.append("\n")
        
        # Recommendations
        content.append("## Recommendations\n")
        for i, rec in enumerate(report_data['recommendations'], 1):
            content.append(f"{i}. {rec}\n")
        
        return "".join(content)
    
    def _generate_fcs_report(self, report_data: Dict[str, Any], 
                           data_info: Dict[str, Any], 
                           analysis_results: Dict[str, Any], 
                           specialized_results: Dict[str, Any]) -> str:
        """Generate FCS-focused report"""
        content = []
        
        content.append("# Fluorescence Correlation Spectroscopy Report\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # FCS Data Information
        content.append("## FCS Data Information\n")
        metadata = report_data['metadata']
        content.append(f"- **Filename:** {metadata['filename']}\n")
        content.append(f"- **Data Type:** FCS Time Series\n")
        content.append(f"- **Time Points:** {metadata['time_points']}\n")
        content.append(f"- **Channels:** {metadata['channels']}\n")
        content.append("\n")
        
        # Correlation Analysis Results
        if analysis_results:
            content.append("## Correlation Analysis Results\n")
            analysis_summary = report_data['analysis_summary']
            
            if 'diffusion_coefficient' in analysis_summary.get('key_findings', {}):
                content.append(f"- **Diffusion Coefficient:** {analysis_summary['key_findings']['diffusion_coefficient']}\n")
            if 'correlation_time' in analysis_summary.get('key_findings', {}):
                content.append(f"- **Correlation Time:** {analysis_summary['key_findings']['correlation_time']}\n")
            
            content.append("\n")
        
        # Advanced Analysis
        if specialized_results:
            content.append("## Advanced Correlation Methods\n")
            specialized_summary = report_data['specialized_summary']
            
            if specialized_summary.get('specialized_methods'):
                content.append(f"- **Methods Applied:** {', '.join(specialized_summary['specialized_methods'])}\n")
            
            if specialized_summary.get('physics_findings'):
                for key, value in specialized_summary['physics_findings'].items():
                    content.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            content.append("\n")
        
        # Biophysical Interpretation
        content.append("## Biophysical Interpretation\n")
        for i, rec in enumerate(report_data['recommendations'], 1):
            content.append(f"{i}. {rec}\n")
        
        return "".join(content)
    
    def _generate_physics_report(self, report_data: Dict[str, Any], 
                               data_info: Dict[str, Any], 
                               analysis_results: Dict[str, Any], 
                               specialized_results: Dict[str, Any]) -> str:
        """Generate specialized physics report"""
        content = []
        
        content.append("# Specialized Physics Analysis Report\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Overview
        content.append("## Dataset Overview\n")
        metadata = report_data['metadata']
        content.append(f"- **Filename:** {metadata['filename']}\n")
        content.append(f"- **System:** {metadata['microscope_type']}\n")
        content.append(f"- **Dataset Size:** {metadata.get('dataset_size', metadata['dimensions'])}\n")
        content.append("\n")
        
        # Specialized Methods Results
        if specialized_results:
            content.append("## Specialized Physics Results\n")
            specialized_summary = report_data['specialized_summary']
            
            # Optical Flow Analysis
            if 'optical_flow' in specialized_results:
                content.append("### Optical Flow & Elastography\n")
                flow_result = specialized_results['optical_flow']
                content.append(f"- **Method:** {flow_result.get('method', 'Unknown')}\n")
                
                if 'avg_displacement' in flow_result:
                    content.append(f"- **Average Displacement:** {flow_result['avg_displacement']:.3f} pixels\n")
                if 'max_displacement' in flow_result:
                    content.append(f"- **Maximum Displacement:** {flow_result['max_displacement']:.3f} pixels\n")
                if 'num_tracked_points' in flow_result:
                    content.append(f"- **Tracked Points:** {flow_result['num_tracked_points']}\n")
                content.append("\n")
            
            # ICS Analysis
            if 'ics' in specialized_results:
                content.append("### Image Correlation Spectroscopy\n")
                ics_result = specialized_results['ics']
                content.append(f"- **Method:** {ics_result.get('method', 'Unknown')}\n")
                
                if 'diffusion_coefficient' in ics_result:
                    content.append(f"- **Diffusion Coefficient:** {ics_result['diffusion_coefficient']:.6f} μm²/s\n")
                if 'flow_magnitude' in ics_result:
                    content.append(f"- **Flow Velocity:** {ics_result['flow_magnitude']:.3f} μm/s\n")
                if 'fit_quality' in ics_result:
                    content.append(f"- **Fit Quality (R²):** {ics_result['fit_quality']:.3f}\n")
                content.append("\n")
        
        # Physical Interpretation
        content.append("## Physical Interpretation\n")
        for i, rec in enumerate(report_data['recommendations'], 1):
            content.append(f"{i}. {rec}\n")
        
        return "".join(content)
    
    def _generate_ai_report(self, report_data: Dict[str, Any], 
                          data_info: Dict[str, Any], 
                          analysis_results: Dict[str, Any], 
                          specialized_results: Dict[str, Any]) -> str:
        """Generate AI enhancement report"""
        content = []
        
        content.append("# AI Enhancement Analysis Report\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Enhancement Overview
        content.append("## AI Enhancement Overview\n")
        metadata = report_data['metadata']
        content.append(f"- **Original Dataset:** {metadata['filename']}\n")
        content.append(f"- **Image Dimensions:** {metadata['dimensions']}\n")
        content.append(f"- **Data Quality:** {report_data['data_summary']['data_quality']}\n")
        content.append("\n")
        
        # AI Methods Applied
        content.append("## AI Enhancement Methods\n")
        content.append("- Methods available: Noise2Void, CARE restoration, Cellpose segmentation, StarDist\n")
        content.append("- Processing optimized for microscopy data characteristics\n")
        content.append("\n")
        
        # Post-Enhancement Analysis
        if analysis_results or specialized_results:
            content.append("## Post-Enhancement Analysis Results\n")
            
            if analysis_results:
                analysis_summary = report_data['analysis_summary']
                content.append(f"- **Standard Methods:** {', '.join(analysis_summary.get('analyses_performed', []))}\n")
            
            if specialized_results:
                specialized_summary = report_data['specialized_summary']
                content.append(f"- **Specialized Methods:** {', '.join(specialized_summary.get('specialized_methods', []))}\n")
            content.append("\n")
        
        # Enhancement Impact
        content.append("## Enhancement Impact & Recommendations\n")
        for i, rec in enumerate(report_data['recommendations'], 1):
            content.append(f"{i}. {rec}\n")
        
        return "".join(content)
    
    def _generate_comprehensive_report(self, report_data: Dict[str, Any], 
                                     data_info: Dict[str, Any], 
                                     analysis_results: Dict[str, Any], 
                                     specialized_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        content = []
        
        content.append("# Comprehensive Biophysics Analysis Report\n")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        content.append("## Executive Summary\n")
        metadata = report_data['metadata']
        content.append(f"This report presents a comprehensive analysis of the microscopy dataset '{metadata['filename']}' ")
        content.append(f"acquired using {metadata['microscope_type']} system. ")
        
        analysis_count = len(report_data['analysis_summary'].get('analyses_performed', []))
        specialized_count = len(report_data['specialized_summary'].get('specialized_methods', []))
        total_methods = analysis_count + specialized_count
        
        content.append(f"A total of {total_methods} analysis methods were applied, ")
        content.append(f"including {analysis_count} standard biophysical techniques and {specialized_count} specialized physics methods.\n\n")
        
        # Dataset Information
        content.append("## Dataset Information\n")
        content.append(f"- **Filename:** {metadata['filename']}\n")
        content.append(f"- **Microscope System:** {metadata['microscope_type']}\n")
        content.append(f"- **Format:** {metadata['format']}\n")
        content.append(f"- **Dimensions:** {metadata['dimensions']}\n")
        content.append(f"- **Pixel Size:** {metadata['pixel_size']} μm\n")
        content.append(f"- **Time Points:** {metadata['time_points']}\n")
        content.append(f"- **Channels:** {metadata['channels']}\n")
        if metadata['channel_names']:
            content.append(f"- **Channel Names:** {', '.join(metadata['channel_names'])}\n")
        content.append(f"- **Acquisition Date:** {metadata['acquisition_date']}\n")
        content.append("\n")
        
        # Data Quality Assessment
        content.append("## Data Quality Assessment\n")
        characteristics = report_data['data_summary']
        content.append(f"- **Overall Quality:** {characteristics['data_quality']}\n")
        content.append(f"- **Spatial Resolution:** {characteristics['spatial_resolution']}\n")
        content.append(f"- **Temporal Resolution:** {characteristics['temporal_resolution']}\n")
        
        if characteristics.get('recommended_analyses'):
            content.append("- **Recommended Analysis Types:**\n")
            for analysis in characteristics['recommended_analyses']:
                content.append(f"  - {analysis}\n")
        content.append("\n")
        
        # Standard Analysis Results
        if analysis_results:
            content.append("## Standard Biophysical Analysis\n")
            analysis_summary = report_data['analysis_summary']
            content.append(f"**Methods Applied:** {', '.join(analysis_summary['analyses_performed'])}\n\n")
            
            if analysis_summary['key_findings']:
                content.append("**Key Findings:**\n")
                for key, value in analysis_summary['key_findings'].items():
                    content.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                content.append("\n")
        
        # Specialized Physics Analysis
        if specialized_results:
            content.append("## Specialized Physics Analysis\n")
            specialized_summary = report_data['specialized_summary']
            
            if specialized_summary.get('specialized_methods'):
                content.append(f"**Methods Applied:** {', '.join(specialized_summary['specialized_methods'])}\n\n")
            
            if specialized_summary.get('physics_findings'):
                content.append("**Physics Findings:**\n")
                for key, value in specialized_summary['physics_findings'].items():
                    content.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                content.append("\n")
            
            if specialized_summary.get('advanced_metrics'):
                content.append("**Advanced Metrics:**\n")
                for key, value in specialized_summary['advanced_metrics'].items():
                    content.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                content.append("\n")
        
        # Conclusions and Recommendations
        content.append("## Conclusions and Recommendations\n")
        if report_data['recommendations']:
            for i, rec in enumerate(report_data['recommendations'], 1):
                content.append(f"{i}. {rec}\n")
        else:
            content.append("No specific recommendations generated. Consider applying additional analysis methods based on experimental objectives.\n")
        
        content.append("\n")
        
        # Technical Details
        content.append("## Technical Details\n")
        content.append(f"- **Analysis Framework:** Advanced Image Biophysics Platform\n")
        content.append(f"- **Report Generation Time:** {report_data['timestamp']}\n")
        content.append(f"- **Total Dataset Size:** {metadata.get('total_pixels', 'Unknown')} pixels\n")
        if metadata.get('file_size'):
            content.append(f"- **File Size:** {metadata['file_size']}\n")
        
        return "".join(content)
    
    def export_report_formats(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Export report in multiple formats"""
        exports = {}
        
        # Text/Markdown format
        exports['markdown'] = report['report_content']
        
        # JSON format
        exports['json'] = json.dumps(report['report_data'], indent=2, default=str)
        
        # CSV format for tabular data
        if report['report_data'].get('analysis_summary', {}).get('key_findings'):
            findings_df = pd.DataFrame(list(report['report_data']['analysis_summary']['key_findings'].items()), 
                                     columns=['Metric', 'Value'])
            exports['csv'] = findings_df.to_csv(index=False)
        
        # HTML format
        html_content = self._convert_markdown_to_html(report['report_content'])
        exports['html'] = html_content
        
        return exports
    
    def _convert_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to basic HTML"""
        html_content = markdown_content
        
        # Convert headers
        html_content = html_content.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
        html_content = html_content.replace('## ', '<h2>').replace('\n', '</h2>\n')
        html_content = html_content.replace('### ', '<h3>').replace('\n', '</h3>\n')
        
        # Convert bold text
        html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
        
        # Convert lists
        lines = html_content.split('\n')
        html_lines = []
        in_list = False
        
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(f'<li>{line.strip()[2:]}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(line)
        
        if in_list:
            html_lines.append('</ul>')
        
        # Wrap in basic HTML structure
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Biophysics Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; }}
                li {{ margin-bottom: 5px; }}
            </style>
        </head>
        <body>
            {chr(10).join(html_lines)}
        </body>
        </html>
        """
        
        return html_template