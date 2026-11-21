# AI-Powered Weed Detection & Comparative Crop Analysis System

## 1. Project Overview

**Project title:** AI-Powered Weed Detection, Monitoring & Comparative Crop Analysis System

**Interface:** Streamlit  
**Language:** Python  
**Primary goal:** Build an interactive agricultural analytics application that analyzes weed infestation across five major crops and, in a later phase, detects weeds from uploaded crop images using computer vision.

The application will combine:
- A data analytics dashboard using CSV data.
- Crop-wise and weed-wise comparisons.
- Infestation severity and field-level analysis.
- Estimated yield-loss visualization.
- Image-based weed detection using a YOLO object-detection model.
- Downloadable analytical reports.

> Important: The initial CSV is synthetic sample data for software testing. Its values are not validated agricultural measurements. Real field data and annotated images are required before making agronomic or operational claims.

---

## 2. Initial Crop Scope

Start with these five crops:

1. Rice (Paddy)
2. Maize
3. Ragi (Finger millet)
4. Cotton
5. Sugarcane

Keep the crop list configurable so crops can be added or replaced later.

---

## 3. Project Objectives

1. Analyze weed infestation across the selected crops.
2. Compare weed density and infestation severity by crop.
3. Identify common weed types associated with each crop.
4. Visualize field-level and time-based infestation patterns.
5. Estimate potential yield loss using clearly documented assumptions or validated data.
6. Upload crop images and detect weeds using a trained object-detection model.
7. Display detection boxes, labels and confidence scores.
8. Export filtered data and analysis results.

---

## 4. Main Application Modules

### Module 1 — Dashboard
Display key performance indicators:
- Total observation records.
- Number of crops covered.
- Total weed observations/detections.
- Number of high-infestation records.
- Average weed count per square metre.
- Average estimated affected area.
- Average estimated yield loss.

Add filters for crop, date range, field, weed type, growth stage and infestation level.

### Module 2 — Image-Based Weed Detection
- Upload JPG, JPEG or PNG images.
- Select the crop, if known.
- Run the trained YOLO model.
- Display the original image and annotated result.
- Show detected weed classes and confidence scores.
- Display detection counts by class.
- Allow the annotated image and detection summary to be downloaded.

The detection model must be trained or fine-tuned on suitable annotated images. Do not treat the sample CSV as image-training data.

### Module 3 — Crop-Wise Analysis
For each crop, show:
- Number of observations.
- Average weed count per m².
- Most frequently recorded weed types.
- Low, medium and high infestation distribution.
- Average affected area percentage.
- Estimated yield-loss distribution.
- Growth-stage comparison.

### Module 4 — Weed Comparison
- Compare weed types across crops.
- Show the most frequently recorded weeds.
- Compare average weed density by weed type.
- Show crop-versus-weed distribution using a heatmap.
- Filter by date, field and crop.

### Module 5 — Infestation Severity
Use documented thresholds to group records into Low, Medium and High categories. The supplied sample dataset uses:
- Low: fewer than 12 weeds/m².
- Medium: 12–29 weeds/m².
- High: 30 or more weeds/m².

These thresholds are only sample-project rules. Replace them with crop- and weed-specific thresholds from reliable agronomic sources before real-world use.

### Module 6 — Yield-Loss Analysis
- Compare estimated yield loss by crop.
- Analyze the relationship between weed density and estimated yield loss.
- Compare yield-loss estimates by growth stage and treatment.
- Clearly label estimates and document the calculation method.

Do not present synthetic estimates as measured yield loss or as a reliable prediction model.

### Module 7 — Treatment Analysis
Compare observations by treatment:
- No treatment.
- Manual weeding.
- Herbicide.
- Mechanical weeding.

Show observation counts and associated infestation/yield-loss statistics. This is descriptive analysis, not proof that one treatment caused a particular outcome. A valid treatment-effect study needs an appropriate experimental design and controls.

### Module 8 — Reports and Data Export
- Download filtered records as CSV.
- Download detection summaries.
- Export chart images where supported.
- Provide a concise report of the selected filters and key metrics.

---

## 5. Recommended Graphs and Visualizations

Use Plotly for interactive charts.

1. Bar chart — average weed density by crop.
2. Pie/donut chart — infestation severity distribution.
3. Bar chart — estimated yield loss by crop.
4. Stacked bar chart — severity distribution by crop.
5. Heatmap — crop versus weed type.
6. Line chart — monthly infestation trend.
7. Bar chart — most frequently observed weed types.
8. Box plot — weed density distribution by crop.
9. Scatter plot — weed density versus estimated yield loss.
10. Bar chart — observations by treatment method.
11. Bar chart — infestation by crop growth stage.
12. Field comparison chart — weed density by field ID.

Every chart should respond to the dashboard filters. Include units, titles and clear labels.

---

## 6. Technology Stack

| Area | Recommended technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Data processing | Pandas, NumPy |
| Interactive charts | Plotly |
| Image processing | OpenCV, Pillow |
| Object detection | Ultralytics YOLO (choose a supported version) |
| Deep learning | PyTorch |
| Dataset annotation | Roboflow or CVAT |
| Initial data storage | CSV |
| Later data storage | SQLite or PostgreSQL |
| Model evaluation | Precision, Recall, mAP and confusion matrix |
| Version control | Git and GitHub |

Check the current compatibility of Python, PyTorch and the selected YOLO package before pinning versions.

---

## 7. Dataset Plan

The project needs two distinct dataset types.

### A. Analytical CSV dataset

Use the supplied file:

`weed_detection_sample_data_500_records.csv`

It contains 500 synthetic observation records and these columns:

| Column | Meaning |
|---|---|
| record_id | Unique observation identifier |
| observation_date | Observation date |
| crop | Crop name |
| field_id | Field identifier |
| crop_growth_stage | Crop growth stage |
| weed_type | Recorded weed category |
| weed_count_per_m2 | Sample weed count per square metre |
| infestation_level | Low, Medium or High |
| estimated_area_affected_pct | Estimated affected area percentage |
| treatment_applied | Recorded treatment category |
| estimated_yield_loss_pct | Synthetic estimated yield-loss percentage |
| model_detection_confidence | Synthetic confidence value |

Use this dataset to build and test filters, calculations, charts, comparisons and CSV exports.

### B. Image dataset for computer vision

Collect images containing the selected crops and their associated weeds. Prefer images from the target growing region and capture different:
- Lighting and weather conditions.
- Camera angles and distances.
- Crop growth stages.
- Soil backgrounds and field conditions.
- Weed sizes and densities.

Annotate weed objects with bounding boxes and class labels using a consistent annotation guide.

Potential places to explore:
- DeepWeeds dataset: https://github.com/AlexOlsen/DeepWeeds
- Plant Seedlings Dataset: https://vision.eng.au.dk/plant-seedlings-dataset/
- Roboflow Universe: https://universe.roboflow.com/
- CVAT annotation tool: https://www.cvat.ai/

These are starting points, not a guarantee that each dataset contains all five target crops or the required weed classes. Inspect licenses, labels, image quality and class coverage before use. You may need to collect and annotate additional local images.

### Suggested image-dataset folder structure

```text
datasets/
└── weed_detection/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

Split data by source field/session where possible, not merely by randomly splitting near-duplicate images. This reduces data leakage and gives a more realistic test.

---

## 8. Proposed Application Architecture

```text
User
 |
 v
Streamlit Application
 |
 +---------------------------+
 |                           |
 v                           v
CSV Upload / Sample Data     Image Upload
 |                           |
 v                           v
Pandas Data Processing       OpenCV / Image Preprocessing
 |                           |
 v                           v
Analytics & Comparisons      Trained YOLO Model
 |                           |
 v                           v
Plotly Charts                Detection Boxes & Counts
 |                           |
 +-------------+-------------+
               |
               v
       Results & Downloads
```

---

## 9. Suggested Project Folder Structure

```text
weed-detection-system/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── weed_detection_sample_data_500_records.csv
│
├── models/
│   └── best.pt
│
├── pages/
│   ├── 1_Crop_Analysis.py
│   ├── 2_Weed_Comparison.py
│   ├── 3_Infestation_Analysis.py
│   ├── 4_Yield_Analysis.py
│   ├── 5_Image_Detection.py
│   └── 6_Reports.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── analytics.py
│   ├── visualizations.py
│   ├── detection.py
│   └── report_generator.py
│
├── datasets/
│   └── weed_detection/
│
└── outputs/
    ├── annotated_images/
    └── reports/
```

Do not commit large datasets, model weights, private field images or generated outputs to Git unless appropriate. Use `.gitignore` and a suitable storage location.

---

## 10. Step-by-Step Implementation Roadmap

### Phase 1 — Environment Setup
1. Install Python and VS Code.
2. Create the project directory.
3. Create and activate a virtual environment.
4. Install Streamlit, Pandas, NumPy and Plotly.
5. Add the sample CSV to the `data/` folder.
6. Run a basic Streamlit application.

Example commands:

```bash
mkdir weed-detection-system
cd weed-detection-system

python -m venv venv

# Windows
venv\Scripts\activate

pip install streamlit pandas numpy plotly pillow opencv-python
```

Create `requirements.txt` after confirming the working package versions.

### Phase 2 — Build the Data Dashboard
1. Load the sample CSV using Pandas.
2. Validate required columns and data types.
3. Handle missing or invalid values.
4. Add sidebar filters.
5. Create KPI cards.
6. Build the crop-wise charts.
7. Build weed and severity comparisons.
8. Add CSV downloads.
9. Test the dashboard with different filters.

**Deliverable:** A working Streamlit analytics dashboard using the sample CSV.

### Phase 3 — Add Crop-Wise Analytics
1. Implement crop-level summary functions.
2. Calculate average weed density.
3. Calculate severity counts and percentages.
4. Compare weed categories by crop.
5. Add growth-stage and field-level analysis.
6. Add documented yield-loss summaries.
7. Validate all calculations against manually checked examples.

**Deliverable:** Interactive crop and weed comparison pages.

### Phase 4 — Collect and Prepare Image Data
1. Identify target weed classes for each crop.
2. Find suitable public datasets and review their licenses.
3. Collect local images where necessary.
4. Annotate weeds using bounding boxes.
5. Review annotation consistency.
6. Split images into training, validation and test sets.
7. Create the YOLO dataset configuration file.

**Deliverable:** A documented, annotated image dataset.

### Phase 5 — Train the Weed Detection Model
1. Select a suitable YOLO model.
2. Configure the dataset paths and class names.
3. Train on the training images.
4. Evaluate on validation images.
5. Tune image size, epochs and augmentation.
6. Evaluate once on the held-out test set.
7. Save the best model weights.
8. Record class-wise precision, recall and mAP.

**Deliverable:** A tested weed-detection model and evaluation report.

### Phase 6 — Integrate AI Detection into Streamlit
1. Load the trained model.
2. Add image upload.
3. Validate uploaded image formats and size.
4. Run inference.
5. Draw bounding boxes and labels.
6. Show class-wise detection counts.
7. Display confidence scores.
8. Allow annotated-image downloads.
9. Handle cases where no weeds are detected.

**Deliverable:** Working image-based weed detection inside the application.

### Phase 7 — Reports, Testing and Refinement
1. Test every page and filter.
2. Test invalid and empty CSV files.
3. Test different image sizes and formats.
4. Verify chart calculations.
5. Check model performance by crop and weed class.
6. Add report downloads.
7. Improve layout and usability.
8. Document limitations and known failure cases.

**Deliverable:** A complete tested application.

### Phase 8 — Deployment
1. Store code in GitHub.
2. Prepare dependencies and model files.
3. Choose a hosting service with enough memory and compute for the model.
4. Configure secrets and file-size limits if needed.
5. Deploy the Streamlit application.
6. Test the deployed version with sample data and images.

**Deliverable:** A deployed demonstration application.

---

## 11. Model Evaluation

Do not evaluate the AI model only by the number of detections.

Track:
- Precision: how many predicted weed detections are correct.
- Recall: how many actual weeds are detected.
- mAP@0.5.
- mAP@0.5:0.95.
- Confusion matrix.
- Per-class performance.
- Inference time per image.

Evaluate separately for each crop and weed class. A strong overall score can hide poor performance on a particular crop or weed.

---

## 12. Functional Acceptance Checklist

### Analytics dashboard
- [ ] Sample CSV loads successfully.
- [ ] Data validation and error messages work.
- [ ] Crop and date filters work.
- [ ] KPI values update with filters.
- [ ] All charts update with filters.
- [ ] Crop comparisons are correct.
- [ ] CSV export works.

### AI image detection
- [ ] Image upload works.
- [ ] Supported image formats are validated.
- [ ] Model loads successfully.
- [ ] Bounding boxes and class labels display.
- [ ] Confidence values display.
- [ ] Detection counts are correct.
- [ ] No-detection case is handled.
- [ ] Annotated image can be downloaded.

### Quality and deployment
- [ ] Synthetic data is clearly labelled.
- [ ] Real test images are kept separate from training data.
- [ ] Model metrics are documented.
- [ ] Limitations are displayed.
- [ ] Application runs locally.
- [ ] Deployment is tested.

---

## 13. Important Limitations and Responsible Use

- Synthetic CSV values are only for testing the application.
- A weed image dataset does not automatically provide field-level weed density per square metre.
- Converting detections into weeds/m² requires a known image area, camera geometry or a calibrated capture setup.
- Yield loss cannot be reliably inferred from weed count alone. It depends on crop, weed species, growth stage, competition duration, environment and management.
- Detection confidence is not the same as agronomic certainty.
- The model should be tested on images from the intended region and field conditions.
- Treatment comparisons in observational data do not prove treatment effectiveness.
- Do not recommend herbicide products or application rates solely from model predictions.

---

## 14. Recommended Development Order

Build in this order to reduce complexity:

1. Streamlit layout and navigation.
2. CSV upload and sample-data loading.
3. Data cleaning and validation.
4. KPI dashboard.
5. Crop-wise analysis.
6. Weed comparison.
7. Severity and yield-loss analysis.
8. CSV/report downloads.
9. Image upload and display.
10. Annotated weed dataset preparation.
11. YOLO model training and evaluation.
12. Streamlit model integration.
13. Testing and deployment.

**Recommended first milestone:** Finish Phases 1–3 using the supplied 500-record sample CSV. Once the dashboard and analytical calculations are working correctly, proceed to image-dataset preparation and AI model training.
