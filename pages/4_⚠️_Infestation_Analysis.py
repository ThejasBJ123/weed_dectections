"""
Module 5: Infestation Severity & Field Monitoring Page
Field risk scoring, high-risk priority alerts, and spatial distribution.
"""
import streamlit as st
import pandas as pd
from src.data_loader import get_active_dataframe
from src.analytics import get_severity_summary, get_field_risk_summary
from src.visualizations import (
    plot_severity_donut,
    plot_field_comparison,
    plot_severity_by_crop_stacked
)

st.set_page_config(page_title="Infestation Severity | Weed Detection System", page_icon="⚠️", layout="wide")

df = get_active_dataframe(st.session_state)
if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to the main page to load data.")
    st.stop()

st.title("⚠️ Weed Infestation Severity & Field Monitoring")
st.markdown("Threshold-based severity analysis, field-level risk ranking, and hotspot identification.")

# Severity Thresholds card
st.info("""
**Agronomic Infestation Classification Rules:**
- **🟢 Low Infestation (< 12 weeds/m²):** Under economic threshold. Routine surveillance sufficient.
- **🟠 Medium Infestation (12 – 29 weeds/m²):** Moderate competition. Cultivation or localized intervention suggested.
- **🔴 High Infestation (≥ 30 weeds/m²):** Severe infestation. Immediate management required to prevent significant crop loss.
""")

# Top metrics
sev_counts = df['infestation_level'].value_counts()
low_c = sev_counts.get('Low', 0)
med_c = sev_counts.get('Medium', 0)
high_c = sev_counts.get('High', 0)
total_c = len(df)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Records Evaluated", total_c)
with k2:
    st.metric("🟢 Low Severity", f"{low_c} ({low_c/total_c*100:.1f}%)")
with k3:
    st.metric("🟠 Medium Severity", f"{med_c} ({med_c/total_c*100:.1f}%)")
with k4:
    st.metric("🔴 High Severity Alert", f"{high_c} ({high_c/total_c*100:.1f}%)")

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    st.plotly_chart(plot_severity_donut(df), use_container_width=True)
with col2:
    st.plotly_chart(plot_severity_by_crop_stacked(df), use_container_width=True)

# Field-level comparison
st.subheader("🌾 Field-by-Field Infestation Analysis")
st.plotly_chart(plot_field_comparison(df), use_container_width=True)

# High Risk Fields Leaderboard
st.subheader("🚨 Field Priority Risk Matrix")
st.caption("Fields sorted by high-severity percentage and average weed density.")

field_risk = get_field_risk_summary(df)

# Highlight high risk rows
def color_risk(val):
    if val == 'High Risk':
        return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;'
    elif val == 'Moderate Risk':
        return 'background-color: #ffe0b2; color: #e65100;'
    else:
        return 'background-color: #c8e6c9; color: #1b5e20;'

styled_table = field_risk.style.applymap(color_risk, subset=['risk_level'])
st.dataframe(styled_table, use_container_width=True)
