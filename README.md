# 🌾 AI-Powered Weed Detection & Comparative Crop Analysis System

An end-to-end interactive agricultural intelligence platform combining **exploratory data analytics** on field scouting data and **real-time computer vision weed detection** using **Ultralytics YOLOv8**.

Built with **Streamlit**, **Python**, **Pandas**, **Plotly**, and **PyTorch/Ultralytics**.

---

## 🚀 Key Features & Modules

### 1. 📈 Executive Dashboard (`app.py`)
- Real-time Key Performance Indicators (KPIs): Total observations, active crops, high infestation counts, mean weed density (weeds/m²), area affected %, and estimated yield loss.
- **AI Computer Vision Performance Scorecard:** Live tracking of mAP@0.5 (88.4%), Precision (89.2%), Recall (86.7%), and F1-Score (87.9%) — all verified above the 85% benchmark.
- Multi-criteria sidebar filters: Filter dynamically by Crop, Date Range, Field Identifier, Weed Species, Growth Stage, Severity Level, and Treatment Applied.
- High-level analytical overview charts with Plotly interactivity.
- Integrated dataset switcher: Toggle between built-in 500-record dataset and custom CSV uploads.

### 2. 📷 AI-Powered Weed Detection (`pages/1_📷_Image_Detection.py`)
- Powered by **Ultralytics YOLOv8** (`models/best.pt`).
- Verified core metrics: **mAP@0.5: 88.4%**, **Precision: 89.2%**, **Recall: 86.7%**, **F1: 87.9%** (all exceeding the >85% accuracy standard).
- Multi-source input mode: Upload your own field photos (`.jpg`, `.jpeg`, `.png`), or select from 200 verified high-accuracy images (>85% detection) or 82 verified clean crop field images (0 weeds).
- Configurable inference parameters: Confidence threshold, IoU NMS threshold, and quadrat frame ground area (m²).
- Outputs side-by-side detection visuals, localized bounding boxes, single-image benchmark comparison tables, density gauges, and instant download of annotated images and detection audit CSVs.

### 3. 📊 Crop-Wise Analysis (`pages/2_📊_Crop_Analysis.py`)
- In-depth evaluation across five benchmark crops: **Rice (Paddy)**, **Maize**, **Ragi (Finger millet)**, **Cotton**, and **Sugarcane**.
- Weed density spread box plots, growth-stage vulnerability comparisons, and severity distribution by crop.

### 4. 🌿 Cross-Crop Weed Comparison (`pages/3_🌿_Weed_Comparison.py`)
- Two-way interactive heatmaps showing weed species vs. host crop interactions (by observation frequency and weed density).
- Weed species ranking, occurrence percentages, and individual species profile drilldowns.

### 5. ⚠️ Infestation Severity & Field Monitoring (`pages/4_⚠️_Infestation_Analysis.py`)
- Threshold classification rules:
  - **🟢 Low:** `< 12 weeds/m²`
  - **🟠 Medium:** `12 – 29 weeds/m²`
  - **🔴 High:** `≥ 30 weeds/m²`
- Field Priority Risk Matrix: Ranks field IDs by high-severity percentage to guide scouting and intervention teams.

### 6. 📉 Estimated Yield-Loss Analysis (`pages/5_📉_Yield_Analysis.py`)
- Empirical yield loss regression models linking weed density with percentage yield penalty.
- Critical Weed-Free Period (CWFP) assessment across crop growth stages.
- Treatment method comparison: Evaluates weed suppression and yield retention across *No Treatment*, *Manual Weeding*, *Herbicide*, and *Mechanical Weeding*.

### 7. 📄 Reports & Model Training Evaluation (`pages/6_📄_Reports.py`)
- Inspect actual YOLO training curves: F1-Confidence, Precision-Recall, Loss curves, and spatial label distribution correlograms.
- Comprehensive Model Performance Scorecard comparing all metrics against the 85% requirement.
- Export filtered records as CSV and JSON.
- Generate and download formal Markdown / HTML Agronomic Executive Reports.

---

## 🎯 AI Model Evaluation Benchmarks (All Core Metrics > 85%)

| Evaluation Metric | Benchmark Result | Minimum Target Requirement | Target Margin | Audit Status |
| :--- | :--- | :--- | :--- | :--- |
| **mAP@0.5** (Mean Average Precision) | **88.4%** | ≥ 85.0% | **+3.4%** | ✅ Exceeds Target |
| **Precision (P)** (Detection Accuracy) | **89.2%** | ≥ 85.0% | **+4.2%** | ✅ Exceeds Target |
| **Recall (R)** (Weed Capture Sensitivity) | **86.7%** | ≥ 85.0% | **+1.7%** | ✅ Exceeds Target |
| **F1-Score** (Precision-Recall Harmonic Mean) | **87.9%** | ≥ 85.0% | **+2.9%** | ✅ Exceeds Target |
| **mAP@0.5:0.95** (Strict Multi-IoU Mean) | **64.5%** | ≥ 50.0% | **+14.5%** | ✅ Exceeds Target |


---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Web Interface** | Streamlit |
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Plotly Express & Plotly Graph Objects |
| **Object Detection** | Ultralytics YOLOv8, PyTorch |
| **Image Processing** | OpenCV, Pillow |
| **Dataset Storage** | CSV (Tabular), YOLO txt format (Images) |

---

## 📁 Project Directory Structure

```text
WEED PROJECT/
├── app.py                                  # Main Streamlit Dashboard & Navigation
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── DATASET_GUIDE.md                        # Complete step-by-step dataset guide
├── .gitignore                              # Git exclusion rules
│
├── data/
│   └── weed_detection_sample_data_500_records.csv  # 500-record analytical dataset
│
├── models/
│   └── best.pt                             # Trained YOLOv8s weed detection weights
│
├── sample_images/                          # Pre-packaged test field images
│   ├── rice_field_sample_1.jpg
│   ├── maize_crop_weeds_2.jpg
│   ├── cotton_field_weeds_3.jpg
│   └── sugarcane_weeds_4.jpg
│
├── evaluation/                             # YOLO training curves & performance charts
│   ├── results.png
│   ├── F1_curve.png
│   ├── PR_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   ├── labels.jpg
│   └── labels_correlogram.jpg
│
├── pages/
│   ├── 1_📊_Crop_Analysis.py
│   ├── 2_🌿_Weed_Comparison.py
│   ├── 3_⚠️_Infestation_Analysis.py
│   ├── 4_📉_Yield_Analysis.py
│   ├── 5_📷_Image_Detection.py
│   └── 6_📄_Reports.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      # Data validation and loading
│   ├── preprocessing.py                    # Multi-filter engine
│   ├── analytics.py                        # KPI & agronomic calculation engine
│   ├── visualizations.py                  # 12 Plotly visualization charts
│   ├── detection.py                        # YOLOv8 inference & bounding box rendering
│   └── report_generator.py                 # Export & executive report generation
│
├── datasets/
│   └── weed_detection/                     # Standard YOLO image dataset layout
│       ├── data.yaml
│       ├── images/ (train, val, test)
│       └── labels/ (train, val, test)
│
└── outputs/
    ├── annotated_images/                   # Saved detection images
    └── reports/                            # Generated audit logs & reports
```

---

## ⚡ Quick Start Guide

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd "WEED PROJECT"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit app from your terminal:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 📂 How to Add Datasets

For full step-by-step instructions with examples and schemas, see [DATASET_GUIDE.md](file:///d:/Thejas/WEED%20PROJECT/DATASET_GUIDE.md).

### Quick Summary:
1. **To add/update tabular CSV records:**
   - Place your CSV in `data/weed_detection_sample_data_500_records.csv` or upload via the web dashboard sidebar.
   - Required columns: `record_id`, `observation_date`, `crop`, `field_id`, `crop_growth_stage`, `weed_type`, `weed_count_per_m2`, `infestation_level`, `estimated_area_affected_pct`, `treatment_applied`, `estimated_yield_loss_pct`, `model_detection_confidence`.
2. **To add images for YOLO detection:**
   - For instant UI testing: Drop images into `sample_images/`.
   - For model training/fine-tuning: Place annotated images into `datasets/weed_detection/images/` and `labels/` using YOLO format and run `python train1.py`.

---

## 📜 Agronomic Disclaimer & Limitations
- Initial CSV values represent analytical synthetic benchmark records for software testing and workflow validation.
- Field weed density per m² from computer vision depends on camera height and calibrated quadrat surface area.
- Yield loss models are empirical indicators; always consult certified agronomists before operational herbicide or chemical treatment applications.
