
def _generate_comprehensive_report(self, report_data: Dict[str, Any], 
                                     data_info: Dict[str, Any], 
                                     analysis_results: Dict[str, Any], 
                                     specialized_results: Dict[str, Any]) -> str:
    """Generate comprehensive analysis report with more detail."""
    content = []

    # ... (header and executive summary - to be improved) ...

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

    # ... (rest of the report) ...
    return "".join(content)
