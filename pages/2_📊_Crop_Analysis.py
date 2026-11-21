"""
Module 3: Crop-Wise Analysis Page
Detailed analysis and comparisons across selected crops.
"""
import streamlit as st
import pandas as pd
from src.data_loader import get_active_dataframe
from src.preprocessing import filter_dataset
from src.analytics import get_crop_summary
from src.visualizations import (
    plot_weed_density_box,
    plot_severity_by_crop_stacked,
    plot_growth_stage_infestation,
    plot_yield_loss_by_crop
)

st.set_page_config(page_title="Crop Analysis | Weed Detection System", page_icon="📊", layout="wide")

df = get_active_dataframe(st.session_state)
if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to the main page to load data.")
    st.stop()

st.title("📊 Crop-Wise Weed Infestation Analysis")
st.markdown("Detailed breakdown of weed density, infestation severity, growth stages, and yield loss for each crop.")

# Sidebar filter for crop selection
available_crops = sorted(df['crop'].unique().tolist())
selected_crop = st.sidebar.selectbox("Select Target Crop for In-Depth Analysis:", ["All Crops"] + available_crops)

if selected_crop != "All Crops":
    crop_df = df[df['crop'] == selected_crop]
else:
    crop_df = df

# Metric cards
crop_summary = get_crop_summary(crop_df)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Observations", len(crop_df))
with c2:
    st.metric("Mean Weed Density", f"{crop_df['weed_count_per_m2'].mean():.1f} /m²")
with c3:
    high_pct = (crop_df['infestation_level'] == 'High').mean() * 100
    st.metric("High Infestation Share", f"{high_pct:.1f}%")
with c4:
    st.metric("Avg Area Affected", f"{crop_df['estimated_area_affected_pct'].mean():.1f}%")
with c5:
    st.metric("Mean Yield Loss", f"{crop_df['estimated_yield_loss_pct'].mean():.1f}%")

st.markdown("---")

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(plot_weed_density_box(crop_df), use_container_width=True)

with col2:
    st.plotly_chart(plot_growth_stage_infestation(crop_df), use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(plot_severity_by_crop_stacked(crop_df), use_container_width=True)

with col4:
    st.plotly_chart(plot_yield_loss_by_crop(crop_df), use_container_width=True)

# Dominant weeds for this crop
st.subheader("🌾 Dominant Weed Species Distribution")
weed_dist = crop_df.groupby('weed_type').agg(
    records=('record_id', 'count'),
    mean_density=('weed_count_per_m2', 'mean'),
    mean_yield_loss=('estimated_yield_loss_pct', 'mean')
).reset_index().sort_values('records', ascending=False)

weed_dist['mean_density'] = weed_dist['mean_density'].round(1)
weed_dist['mean_yield_loss'] = weed_dist['mean_yield_loss'].round(1)

st.dataframe(weed_dist, use_container_width=True)

# Raw records table
with st.expander("🔍 View Raw Observation Records"):
    st.dataframe(crop_df.sort_values('observation_date', ascending=False), use_container_width=True)
