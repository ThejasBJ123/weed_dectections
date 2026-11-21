"""
Module 8 & Model Evaluation: Reports & Model Evaluation Metrics
Model performance graphs (mAP, F1, Loss) and data export suite.
"""
import os
import streamlit as st
import pandas as pd
from PIL import Image

from src.data_loader import get_active_dataframe
from src.analytics import get_kpi_metrics
from src.report_generator import (
    export_dataframe_to_csv,
    export_dataframe_to_json,
    generate_executive_report_md
)

st.set_page_config(page_title="Reports & Model Metrics | Weed Detection System", page_icon="📄", layout="wide")

df = get_active_dataframe(st.session_state)
kpis = get_kpi_metrics(df)

st.title("📄 Analytical Reports & Model Evaluation Suite")
st.markdown("Download field reports and inspect trained YOLO model evaluation curves.")

tab1, tab2 = st.tabs(["🤖 YOLO Model Training & Evaluation Metrics", "📥 Export Reports & Datasets"])

with tab1:
    st.subheader("🎯 YOLOv8 Object Detection Model Performance")
    st.markdown("""
    Evaluation metrics generated during the training and validation of the weed detection model (`models/best.pt`).
    - **Architecture:** Ultralytics YOLOv8s
    - **Target Class:** Weed (`0: weed`)
    - **Input Resolution:** 1280x1280
    """)

    eval_dir = os.path.join(os.getcwd(), 'evaluation')
    
    # Display Results curve
    results_path = os.path.join(eval_dir, 'results.png')
    if os.path.exists(results_path):
        st.markdown("#### 1. Training & Validation Curves (Loss, Precision, Recall, mAP@0.5, mAP@0.5:0.95)")
        st.image(results_path, use_container_width=True)

    # Curves Grid
    c_col1, c_col2 = st.columns(2)
    f1_path = os.path.join(eval_dir, 'F1_curve.png')
    pr_path = os.path.join(eval_dir, 'PR_curve.png')

    with c_col1:
        if os.path.exists(f1_path):
            st.markdown("#### F1-Confidence Curve")
            st.image(f1_path, use_container_width=True)
            st.caption("F1 curve illustrates harmonic mean between precision and recall across confidence thresholds.")

    with c_col2:
        if os.path.exists(pr_path):
            st.markdown("#### Precision-Recall (PR) Curve")
            st.image(pr_path, use_container_width=True)
            st.caption("PR curve quantifies detector precision over recall levels (Area under curve = mAP@0.5).")

    p_path = os.path.join(eval_dir, 'P_curve.png')
    r_path = os.path.join(eval_dir, 'R_curve.png')
    c_col3, c_col4 = st.columns(2)

    with c_col3:
        if os.path.exists(p_path):
            st.markdown("#### Precision-Confidence Curve")
            st.image(p_path, use_container_width=True)

    with c_col4:
        if os.path.exists(r_path):
            st.markdown("#### Recall-Confidence Curve")
            st.image(r_path, use_container_width=True)

    labels_path = os.path.join(eval_dir, 'labels.jpg')
    correlogram_path = os.path.join(eval_dir, 'labels_correlogram.jpg')
    if os.path.exists(labels_path) or os.path.exists(correlogram_path):
        st.markdown("#### Dataset Label Distribution & Bounding Box Spatial Correlogram")
        lbl_c1, lbl_c2 = st.columns(2)
        with lbl_c1:
            if os.path.exists(labels_path):
                st.image(labels_path, caption="Bounding Box Instances and Spatial Positions", use_container_width=True)
        with lbl_c2:
            if os.path.exists(correlogram_path):
                st.image(correlogram_path, caption="Box Dimensions & Aspect Ratio Correlogram", use_container_width=True)

with tab2:
    st.subheader("📊 Export Observation Records & Executive Reports")
    
    # Download buttons
    d1, d2 = st.columns(2)
    with d1:
        csv_data = export_dataframe_to_csv(df)
        st.download_button(
            label="💾 Download Observation Records (CSV)",
            data=csv_data,
            file_name="weed_infestation_records_export.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        json_data = export_dataframe_to_json(df)
        st.download_button(
            label="💾 Download Observation Records (JSON)",
            data=json_data,
            file_name="weed_infestation_records_export.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📑 Agronomic Executive Report Preview")
    
    report_md = generate_executive_report_md(df, kpis)
    st.markdown(report_md)

    st.download_button(
        label="📥 Download Executive Report (Markdown .md)",
        data=report_md.encode('utf-8'),
        file_name="Agronomic_Weed_Infestation_Report.md",
        mime="text/markdown",
        use_container_width=True
    )
