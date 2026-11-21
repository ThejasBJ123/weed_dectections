# 🌾 Complete Guide: How to Add & Update Datasets

This project uses **two distinct types of datasets**:
1. **Analytical Tabular Dataset (CSV)** — Powers the Streamlit KPI dashboards, crop-wise charts, weed comparisons, infestation severity analysis, and yield loss projections.
2. **Computer Vision Image Dataset (YOLO Format)** — Powers the AI weed detection model (`YOLOv8`) for automated bounding box detection and weed localization.

Below is the step-by-step guide to add, format, and integrate both datasets.

---

## 📂 Part 1: How to Add or Update the Tabular CSV Dataset

### Step 1.1: File Location & Naming
- The default dataset is stored at:
  ```text
  data/weed_detection_sample_data_500_records.csv
  ```
- You can either **replace** this file or **upload your own CSV** directly inside the Streamlit web application using the **Dataset Manager** in the sidebar.

### Step 1.2: Required CSV Columns Schema
Your CSV must contain the following **12 columns** (exact spelling is required):

| Column Name | Data Type | Description | Accepted Values / Examples |
|---|---|---|---|
| `record_id` | String | Unique identifier for observation | `REC-001`, `REC-002`, `OBS_104` |
| `observation_date` | Date / String | Date of field scouting | `YYYY-MM-DD` (e.g. `2026-05-14`) |
| `crop` | String | Target crop name | `Rice (Paddy)`, `Maize`, `Ragi (Finger millet)`, `Cotton`, `Sugarcane` |
| `field_id` | String | Field identifier | `FLD-01`, `FLD-02`, `Plot-A` |
| `crop_growth_stage` | String | Phenological growth stage | `Seedling`, `Vegetative`, `Flowering`, `Maturity` |
| `weed_type` | String | Common weed species or class | `Barnyard Grass`, `Nutgrass`, `Parthenium`, `Bermuda Grass`, `Broadleaf Weeds` |
| `weed_count_per_m2` | Integer / Float | Weeds counted per square metre | Numeric e.g., `14`, `32`, `5` |
| `infestation_level` | String | Severity rating | `Low` (<12), `Medium` (12–29), `High` (≥30) |
| `estimated_area_affected_pct` | Float | Estimated field coverage % | `5.0` to `90.0` |
| `treatment_applied` | String | Weed management applied | `No treatment`, `Manual weeding`, `Herbicide`, `Mechanical weeding` |
| `estimated_yield_loss_pct` | Float | Estimated percentage loss | `2.5` to `50.0` |
| `model_detection_confidence` | Float | AI model score or field certainty | `0.70` to `0.99` |

### Step 1.3: Verification
Once you place your CSV file in `data/` or upload via the web dashboard, the app automatically validates:
- Presence of all required columns.
- Correct numeric parsing and date formatting.
- Auto-categorization of severity levels based on the standard rule:
  - **Low:** `< 12 weeds/m²`
  - **Medium:** `12 – 29 weeds/m²`
  - **High:** `≥ 30 weeds/m²`

---

## 📷 Part 2: How to Add Image Datasets for YOLOv8 Object Detection

### Step 2.1: Image Dataset Directory Structure
Place your raw and annotated images in the `datasets/weed_detection/` directory following the standard YOLO format:

```text
datasets/
└── weed_detection/
    ├── data.yaml
    ├── images/
    │   ├── train/     # 70% of training photos (.jpg, .jpeg, .png)
    │   ├── val/       # 20% of validation photos (.jpg, .jpeg, .png)
    │   └── test/      # 10% of held-out test photos (.jpg, .jpeg, .png)
    └── labels/
        ├── train/     # Corresponding YOLO text labels (.txt)
        ├── val/       # Corresponding YOLO text labels (.txt)
        └── test/      # Corresponding YOLO text labels (.txt)
```

### Step 2.2: Image Collection Best Practices
- Capture crop fields under natural lighting (morning, noon, overcast).
- Vary camera angles and ground-to-camera heights (ideal: 0.8m to 1.5m pointing downwards).
- Include multiple crop growth stages and weed densities.

### Step 2.3: Image Annotation (Labeling)
You can annotate weed objects with bounding boxes using either:
- **CVAT (Computer Vision Annotation Tool):** [https://www.cvat.ai/](https://www.cvat.ai/)
- **Roboflow:** [https://universe.roboflow.com/](https://universe.roboflow.com/)
- **LabelImg:** Local python desktop labeling tool.

#### YOLO Label Format (.txt)
For every image (e.g. `field_001.jpg`), there must be a matching label file (e.g. `field_001.txt`).  
Each line represents one weed bounding box normalized between `0.0` and `1.0`:
```text
<class_id> <x_center> <y_center> <width> <height>
```
*Example (`field_001.txt`):*
```text
0 0.5300 0.5800 0.0400 0.0600
0 0.3800 0.9300 0.0900 0.0900
```
*(Where `0` is the class index for `weed`).*

### Step 2.4: Dataset Configuration (`data.yaml`)
Create or edit `datasets/weed_detection/data.yaml`:
```yaml
path: datasets/weed_detection
train: images/train
val: images/val
test: images/test

names:
  0: weed
```

### Step 2.5: Train or Fine-Tune YOLOv8
Run the included training script (`train1.py`) to train the model on your new dataset:

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8s.pt")
    
    # Train on your dataset
    model.train(
        data="datasets/weed_detection/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=4,
        patience=10,
        project="runs_weed",
        name="weed_model",
        save=True
    )
```

Run training from terminal:
```bash
python train1.py
```

### Step 2.6: Update the Model Weights in Streamlit
When training finishes, your best weights are saved in:
```text
runs_weed/weed_model/weights/best.pt
```
Copy these weights into the application model folder:
```bash
copy runs_weed\weed_model\weights\best.pt models\best.pt
```
Restart the Streamlit application. The **AI Weed Detection** page will immediately use your newly trained model!

---

## 🧪 Part 3: Instant Testing with Sample Images
If you want to quickly test the AI detector without collecting a full dataset:
1. Place any crop/weed field photo into the `sample_images/` directory:
   ```text
   sample_images/
   ├── rice_field_sample_1.jpg
   ├── maize_crop_weeds_2.jpg
   ├── cotton_field_weeds_3.jpg
   └── sugarcane_weeds_4.jpg
   ```
2. Open the **📷 Image Detection** page in the Streamlit web application.
3. Select your image from the **"Select Sample Image"** dropdown or upload it directly.
4. Click **"Run Weed Detection"** to inspect bounding boxes, weed counts, and infestation severity in real time!
