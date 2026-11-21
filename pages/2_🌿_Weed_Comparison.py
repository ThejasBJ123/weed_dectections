"""
Module 4: Weed Comparison Page
Cross-crop species heatmaps, dominance rankings, and co-occurrence analysis.
"""
import streamlit as st
import pandas as pd
from src.data_loader import get_active_dataframe
from src.analytics import get_weed_summary, get_crop_weed_matrix
from src.visualizations import plot_crop_weed_heatmap, plot_frequent_weeds_bar

st.set_page_config(page_title="Weed Comparison | Weed Detection System", page_icon="🌿", layout="wide")

df = get_active_dataframe(st.session_state)
if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to the main page to load data.")
    st.stop()

st.title("🌿 Cross-Crop Weed Comparison")
st.markdown("Compare weed species occurrence, density, and host-crop specificity across fields.")

# Top metrics
weed_sum = get_weed_summary(df)
top_weed = weed_sum.iloc[0]['weed_type'] if not weed_sum.empty else "N/A"
top_count = weed_sum.iloc[0]['occurrence_count'] if not weed_sum.empty else 0

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Weed Species Monitored", df['weed_type'].nunique())
with m2:
    st.metric("Most Prevalent Species", top_weed.split('(')[0].strip())
with m3:
    st.metric("Max Recorded Observations", f"{top_count} records")

st.markdown("---")

# Heatmap section
st.subheader("🗺️ Crop vs. Weed Type Interaction Matrix")
matrix_mode = st.radio(
    "Heatmap Display Metric:",
    ["Observation Frequency (Count)", "Average Weed Density (weeds/m²)"],
    horizontal=True
)

metric_key = 'density' if "Density" in matrix_mode else 'count'
st.plotly_chart(plot_crop_weed_heatmap(df, metric=metric_key), use_container_width=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.plotly_chart(plot_frequent_weeds_bar(df, top_n=8), use_container_width=True)

with col2:
    st.subheader("📋 Weed Species Ranking")
    st.dataframe(
        weed_sum[['weed_type', 'occurrence_count', 'occurrence_pct', 'avg_weed_density', 'primary_crop']],
        use_container_width=True,
        height=380
    )

# Individual Weed Drilldown
st.markdown("---")
st.subheader("🔎 Individual Weed Species Profile")
selected_weed = st.selectbox("Select a Weed Species to Inspect:", weed_sum['weed_type'].tolist())

specific_df = df[df['weed_type'] == selected_weed]
sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric("Total Records", len(specific_df))
with sc2:
    st.metric("Average Density", f"{specific_df['weed_count_per_m2'].mean():.1f} /m²")
with sc3:
    st.metric("Associated Yield Loss", f"{specific_df['estimated_yield_loss_pct'].mean():.1f}%")

st.markdown("##### Distribution across crops for this weed:")
crop_split = specific_df.groupby('crop').agg(
    records=('record_id', 'count'),
    mean_density=('weed_count_per_m2', 'mean'),
    max_density=('weed_count_per_m2', 'max')
).reset_index()
st.dataframe(crop_split, use_container_width=True)
