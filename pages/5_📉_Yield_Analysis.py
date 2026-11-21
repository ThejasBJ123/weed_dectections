"""
Module 6 & 7: Yield-Loss Analysis and Treatment Efficacy
Evaluates potential yield reduction based on weed density, growth stages,
and weed management treatments.
"""
import streamlit as st
import pandas as pd
from src.data_loader import get_active_dataframe
from src.analytics import get_treatment_summary
from src.visualizations import (
    plot_yield_loss_by_crop,
    plot_density_vs_yield_scatter,
    plot_treatment_comparison
)

st.set_page_config(page_title="Yield Loss & Treatment | Weed Detection System", page_icon="📉", layout="wide")

df = get_active_dataframe(st.session_state)
if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to the main page to load data.")
    st.stop()

st.title("📉 Estimated Yield Loss & Treatment Analysis")
st.markdown("Investigate the empirical relationship between weed density, crop growth stage, management treatments, and potential yield loss.")

with st.expander("ℹ️ Agronomic Note on Yield-Loss Estimation", expanded=True):
    st.markdown("""
    **Analytical Context:**
    - Crop yield loss from weed competition depends strongly on the **Critical Weed-Free Period (CWFP)**. Competition in early stages (Seedling and Vegetative) causes significantly steeper yield reductions than competition near maturity.
    - Observational treatment comparisons reflect recorded field management conditions; experimental randomized trials are required to establish causal efficacy.
    """)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Overall Mean Yield Loss", f"{df['estimated_yield_loss_pct'].mean():.1f}%")
with m2:
    max_loss_crop = df.groupby('crop')['estimated_yield_loss_pct'].mean().idxmax()
    st.metric("Highest Impact Crop", max_loss_crop)
with m3:
    untreated_loss = df[df['treatment_applied'] == 'No treatment']['estimated_yield_loss_pct'].mean()
    st.metric("Untreated Field Loss", f"{untreated_loss:.1f}%")
with m4:
    herbicide_loss = df[df['treatment_applied'] == 'Herbicide']['estimated_yield_loss_pct'].mean()
    st.metric("Herbicide Field Loss", f"{herbicide_loss:.1f}%")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_density_vs_yield_scatter(df), use_container_width=True)
with col2:
    st.plotly_chart(plot_yield_loss_by_crop(df), use_container_width=True)

# Treatment comparison
st.subheader("🛠️ Weed Treatment Method Comparison")
st.markdown("Comparing weed suppression and yield retention across different intervention strategies.")

col3, col4 = st.columns([3, 2])
with col3:
    st.plotly_chart(plot_treatment_comparison(df), use_container_width=True)
with col4:
    st.markdown("##### Treatment Statistics Breakdown")
    treatment_df = get_treatment_summary(df)
    st.dataframe(treatment_df, use_container_width=True, height=280)

# Growth stage yield loss impact
st.subheader("🌱 Growth Stage Vulnerability Analysis")
stage_loss = df.groupby('crop_growth_stage').agg(
    observations=('record_id', 'count'),
    mean_density=('weed_count_per_m2', 'mean'),
    mean_yield_loss=('estimated_yield_loss_pct', 'mean')
).reindex(['Seedling', 'Vegetative', 'Flowering', 'Maturity']).reset_index()

stage_loss['mean_density'] = stage_loss['mean_density'].round(1)
stage_loss['mean_yield_loss'] = stage_loss['mean_yield_loss'].round(1)

st.dataframe(stage_loss, use_container_width=True)
st.caption("Notice: Weeds during the Seedling stage exert the highest yield penalty per weed due to root competition and shading.")
