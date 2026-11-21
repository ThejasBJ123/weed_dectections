"""
Report generation and analytical data export module.
Supports CSV, Excel, JSON exports and comprehensive agronomic summary reports.
"""
import io
import json
import datetime
import pandas as pd


def export_dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """
    Converts DataFrame to UTF-8 CSV bytes buffer.
    """
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def export_dataframe_to_json(df: pd.DataFrame) -> bytes:
    """
    Converts DataFrame to JSON bytes.
    """
    return df.to_json(orient='records', date_format='iso', indent=2).encode('utf-8')


def export_detection_summary_csv(detections: list, summary: dict) -> bytes:
    """
    Exports bounding box detections and summary parameters to CSV bytes.
    """
    det_df = pd.DataFrame(detections)
    if det_df.empty:
        det_df = pd.DataFrame(columns=['id', 'class', 'confidence', 'box', 'box_width', 'box_height'])
    
    # Prepend summary metadata as comments or export alongside
    output = io.StringIO()
    output.write(f"# Weed Detection Audit Log\n")
    output.write(f"# Timestamp: {summary.get('timestamp', '')}\n")
    output.write(f"# Total Detections: {summary.get('total_detections', 0)}\n")
    output.write(f"# Estimated Density: {summary.get('estimated_weed_density_m2', 0)} weeds/m2\n")
    output.write(f"# Severity Level: {summary.get('infestation_severity', 'N/A')}\n")
    output.write(f"# Avg Confidence: {summary.get('avg_confidence', 0)}\n")
    output.write(f"# Inference Time: {summary.get('inference_time_ms', 0)} ms\n\n")
    det_df.to_csv(output, index=False)
    return output.getvalue().encode('utf-8')


def generate_executive_report_md(df: pd.DataFrame, kpis: dict, filter_info: str = "All records") -> str:
    """
    Generates a formal Agronomic Weed Monitoring & Crop Impact Report in Markdown.
    """
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
    
    # Crop breakdown table
    crop_stats = df.groupby('crop').agg(
        records=('record_id', 'count'),
        avg_density=('weed_count_per_m2', 'mean'),
        avg_yield_loss=('estimated_yield_loss_pct', 'mean'),
        high_severity=('infestation_level', lambda x: (x == 'High').sum())
    ).reset_index()

    crop_rows = ""
    for _, row in crop_stats.iterrows():
        crop_rows += f"| {row['crop']} | {row['records']} | {row['avg_density']:.1f} weeds/m² | {row['avg_yield_loss']:.1f}% | {row['high_severity']} |\n"

    # Weed breakdown
    weed_counts = df['weed_type'].value_counts().head(5)
    weed_rows = ""
    for w_name, count in weed_counts.items():
        weed_rows += f"- **{w_name}**: {count} observation records ({(count/len(df)*100):.1f}%)\n"

    md = f"""# AI-Powered Weed Infestation & Crop Impact Report
**Generated on:** {now_str}  
**Active Filter Context:** {filter_info}  

---

## 1. Executive Summary & KPIs
- **Total Monitored Observations:** {kpis.get('total_records', 0):,} records
- **Total Crops Monitored:** {kpis.get('num_crops', 0)} crops
- **Overall Mean Weed Density:** **{kpis.get('avg_weed_density', 0.0)} weeds/m²**
- **High-Infestation Critical Records:** {kpis.get('high_infestation_count', 0)} ({kpis.get('high_infestation_pct', 0.0)}%)
- **Average Estimated Area Affected:** {kpis.get('avg_affected_area', 0.0)}%
- **Average Estimated Crop Yield Loss:** **{kpis.get('avg_yield_loss', 0.0)}%**
- **Predominant Weed Species:** {kpis.get('dominant_weed', 'N/A')}
- **Most Impacted Crop:** {kpis.get('most_affected_crop', 'N/A')}

---

## 2. Crop-Wise Infestation Summary

| Crop Name | Records | Mean Weed Density | Mean Yield Loss (%) | High Infestation Alerts |
|---|---|---|---|---|
{crop_rows}

---

## 3. Dominant Weed Species Observed
{weed_rows}

---

## 4. Agronomic Severity Classification Standard
- **Low Severity (< 12 weeds/m²):** Routine monitoring; minimal competition impact.
- **Medium Severity (12–29 weeds/m²):** Interventions recommended (targeted cultivation, localized herbicide).
- **High Severity (≥ 30 weeds/m²):** Immediate weed management required to prevent significant yield penalty.

---

## 5. Agronomic Recommendations
1. Prioritize immediate intervention on fields flagged with **High Severity** and those in **Seedling / Early Vegetative** stages.
2. Formulate integrated weed management (IWM) combining mechanical cultivation and targeted herbicide applications to break resistance.
3. Validate AI detection scores against visual ground-truth sampling quadrat measurements.

---
*Disclaimer: Estimated yield losses in this report represent analytical models derived from weed density benchmarks. Validate with local regional agricultural extension guidance.*
"""
    return md
