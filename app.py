"""
AI-Powered Weed Detection & Comparative Crop Analysis System
Main Application Dashboard (app.py)
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Weed Detection & Crop Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b6b50;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #f9fbf9 0%, #eef7ee 100%);
        border: 1px solid #c8e6c9;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #555;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #777;
    }
    .nav-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 14px;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .nav-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border-color: #81c784;
    }
    .nav-card h4 {
        margin: 0 0 6px 0;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

from src.data_loader import load_dataset, get_active_dataframe
from src.preprocessing import filter_dataset, get_filter_options
from src.analytics import get_kpi_metrics
from src.visualizations import (
    plot_weed_density_by_crop,
    plot_severity_donut,
    plot_yield_loss_by_crop,
    plot_severity_by_crop_stacked,
    plot_infestation_trend,
    plot_frequent_weeds_bar
)

# ----------------- SESSION STATE & DATA MANAGEMENT ----------------- #
if 'dataset_df' not in st.session_state:
    raw_df, is_valid, load_issues = load_dataset()
    if is_valid:
        st.session_state['dataset_df'] = raw_df
        st.session_state['dataset_name'] = "weed_detection_sample_data_500_records.csv"
    else:
        st.session_state['dataset_df'] = pd.DataFrame()
        st.session_state['dataset_name'] = "Failed to load default dataset"

# ----------------- SIDEBAR: DATASET UPLOAD & FILTERS ----------------- #
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=64)
    st.markdown("### 🌾 Weed Analytics System")
    st.caption("AI-Powered Agricultural Monitoring Suite")
    st.markdown("---")

    # Dataset Manager
    st.markdown("#### 📂 Dataset Manager")
    dataset_source = st.radio(
        "Active Dataset Source:",
        ["Built-in Sample (500 records)", "Upload Custom CSV"],
        index=0 if "sample" in st.session_state.get('dataset_name', '') else 1
    )

    if dataset_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV Record File", type=['csv'], key="uploader")
        if uploaded_file is not None:
            user_df, user_valid, user_issues = load_dataset(uploaded_file)
            if user_valid:
                st.session_state['dataset_df'] = user_df
                st.session_state['dataset_name'] = uploaded_file.name
                st.success(f"Loaded: {uploaded_file.name} ({len(user_df)} records)")
            else:
                st.error("Invalid CSV structure!")
                for issue in user_issues:
                    st.warning(f"- {issue}")
    else:
        # Revert to default if changed back
        if "sample" not in st.session_state.get('dataset_name', ''):
            raw_df, _, _ = load_dataset()
            st.session_state['dataset_df'] = raw_df
            st.session_state['dataset_name'] = "weed_detection_sample_data_500_records.csv"
            st.rerun()

    st.info(f"**Current Data:** `{st.session_state.get('dataset_name', 'None')}`")
    st.markdown("---")

    # Filter Section
    st.markdown("#### 🔍 Observation Filters")
    base_df = st.session_state['dataset_df']
    options = get_filter_options(base_df)

    selected_crops = st.multiselect("Select Crops", options['crops'], default=None, placeholder="All Crops")
    
    if options['min_date'] and options['max_date']:
        selected_date_range = st.date_input(
            "Observation Date Range",
            value=(options['min_date'], options['max_date']),
            min_value=options['min_date'],
            max_value=options['max_date']
        )
    else:
        selected_date_range = None

    selected_fields = st.multiselect("Field Identifiers", options['fields'], default=None, placeholder="All Fields")
    selected_weeds = st.multiselect("Weed Species", options['weed_types'], default=None, placeholder="All Weed Types")
    selected_stages = st.multiselect("Crop Growth Stages", options['growth_stages'], default=None, placeholder="All Stages")
    selected_severity = st.multiselect("Infestation Severity", options['infestation_levels'], default=None, placeholder="All Severities")
    selected_treatments = st.multiselect("Applied Treatment", options['treatments'], default=None, placeholder="All Treatments")

    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

# Apply filters
filtered_df = filter_dataset(
    base_df,
    crops=selected_crops,
    date_range=selected_date_range if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2 else None,
    fields=selected_fields,
    weed_types=selected_weeds,
    growth_stages=selected_stages,
    infestation_levels=selected_severity,
    treatments=selected_treatments
)

# ----------------- MAIN CONTENT: HEADER ----------------- #
st.markdown('<div class="main-title">🌿 Weed Detection & Crop Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-crop comparative infestation monitoring, yield-loss estimation & AI computer vision weed detection</div>', unsafe_allow_html=True)

# Important disclaimer alert box
with st.expander("ℹ️ Project Scope, Agronomic Assumptions & Guidelines", expanded=False):
    st.markdown("""
    - **Current Crop Scope:** Rice (Paddy), Maize, Ragi (Finger millet), Cotton, and Sugarcane.
    - **Severity Classification Rule:** Low (< 12 weeds/m²), Medium (12–29 weeds/m²), High (≥ 30 weeds/m²).
    - **Yield Loss Estimation:** Synthetic analytical benchmark models correlate weed density and crop growth stages with potential yield impact.
    - **AI Computer Vision Model:** YOLOv8 trained object detection located at `models/best.pt`.
    """)

if filtered_df.empty:
    st.warning("⚠️ No observation records match the selected filter criteria. Please broaden your filter parameters.")
    st.stop()

# ----------------- KPI METRICS OVERVIEW ----------------- #
kpis = get_kpi_metrics(filtered_df)

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Records</div>
        <div class="kpi-value">{kpis['total_records']}</div>
        <div class="kpi-sub">{kpis['num_crops']} crops included</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Mean Density</div>
        <div class="kpi-value">{kpis['avg_weed_density']}</div>
        <div class="kpi-sub">weeds / m²</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">High Infestation</div>
        <div class="kpi-value" style="color: {'#d62728' if kpis['high_infestation_pct'] > 20 else '#2e7d32'};">{kpis['high_infestation_count']}</div>
        <div class="kpi-sub">{kpis['high_infestation_pct']}% of records</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Area Affected</div>
        <div class="kpi-value">{kpis['avg_affected_area']}%</div>
        <div class="kpi-sub">average field coverage</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Est. Yield Loss</div>
        <div class="kpi-value" style="color: {'#d62728' if kpis['avg_yield_loss'] > 25 else '#f57c00'};">{kpis['avg_yield_loss']}%</div>
        <div class="kpi-sub">potential loss</div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Dominant Weed</div>
        <div class="kpi-value" style="font-size: 1.15rem; word-break: break-word;">{kpis['dominant_weed'].split('(')[0].strip()}</div>
        <div class="kpi-sub">top recorded species</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- QUICK NAVIGATION TILES ----------------- #
st.markdown("### 🧭 Application Modules")
n1, n2, n3, n4, n5, n6 = st.columns(6)

with n1:
    st.markdown("""
    <div class="nav-card">
        <h4>📊 Crop Analysis</h4>
        <p style="font-size: 0.85rem; color: #666;">In-depth crop-wise density, growth stage impact & yield loss.</p>
        <a href="Crop_Analysis" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

with n2:
    st.markdown("""
    <div class="nav-card">
        <h4>🌿 Weed Comparison</h4>
        <p style="font-size: 0.85rem; color: #666;">Cross-crop species heatmaps, dominance rankings & density.</p>
        <a href="Weed_Comparison" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

with n3:
    st.markdown("""
    <div class="nav-card">
        <h4>⚠️ Infestation Risk</h4>
        <p style="font-size: 0.85rem; color: #666;">Severity thresholds, high-risk field matrix & prioritization.</p>
        <a href="Infestation_Analysis" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

with n4:
    st.markdown("""
    <div class="nav-card">
        <h4>📉 Yield Loss</h4>
        <p style="font-size: 0.85rem; color: #666;">Empirical yield loss regressions & treatment efficacy checks.</p>
        <a href="Yield_Analysis" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

with n5:
    st.markdown("""
    <div class="nav-card">
        <h4>📷 AI Detection</h4>
        <p style="font-size: 0.85rem; color: #666;">Upload crop photos & run YOLOv8 object detection model.</p>
        <a href="Image_Detection" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

with n6:
    st.markdown("""
    <div class="nav-card">
        <h4>📄 Reports & Model</h4>
        <p style="font-size: 0.85rem; color: #666;">Download CSV/Reports & view YOLO training evaluation curves.</p>
        <a href="Reports" target="_self" style="font-size: 0.85rem; font-weight: 600; color: #2e7d32;">Open Module →</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- VISUAL ANALYTICS OVERVIEW ----------------- #
st.markdown("### 📈 Analytical Highlights")

row1_col1, row1_col2 = st.columns([3, 2])
with row1_col1:
    st.plotly_chart(plot_weed_density_by_crop(filtered_df), use_container_width=True)
with row1_col2:
    st.plotly_chart(plot_severity_donut(filtered_df), use_container_width=True)

row2_col1, row2_col2 = st.columns([3, 2])
with row2_col1:
    st.plotly_chart(plot_infestation_trend(filtered_df), use_container_width=True)
with row2_col2:
    st.plotly_chart(plot_frequent_weeds_bar(filtered_df, top_n=6), use_container_width=True)

row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    st.plotly_chart(plot_severity_by_crop_stacked(filtered_df), use_container_width=True)
with row3_col2:
    st.plotly_chart(plot_yield_loss_by_crop(filtered_df), use_container_width=True)

# ----------------- RECENT OBSERVATIONS PREVIEW ----------------- #
st.markdown("### 📋 Filtered Records Explorer")
preview_cols = ['record_id', 'observation_date', 'crop', 'field_id', 'crop_growth_stage',
                'weed_type', 'weed_count_per_m2', 'infestation_level', 'estimated_area_affected_pct',
                'treatment_applied', 'estimated_yield_loss_pct']

st.dataframe(
    filtered_df[preview_cols].sort_values('observation_date', ascending=False),
    use_container_width=True,
    height=280
)

st.caption(f"Showing {len(filtered_df)} of {len(base_df)} total records | Data updated for 2026 agricultural season.")
