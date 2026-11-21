"""
YOLOv8 Model Training Script for Weed Detection
"""
import os
import shutil

# Ensure sandbox compatibility on Windows
os.environ.setdefault('YOLO_CONFIG_DIR', os.path.abspath('.ultralytics'))

from ultralytics import YOLO

if __name__ == '__main__':
    data_yaml_path = os.path.abspath("datasets/weed_detection/data.yaml")
    if not os.path.exists(data_yaml_path):
        data_yaml_path = os.path.abspath("data.yaml")

    print(f"Loading YOLOv8 base model...")
    model = YOLO("yolov8s.pt")
    
    print(f"Starting training with dataset: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=1280,
        batch=4,
        workers=0,
        patience=10,
        project="runs_weed",
        name="weed_run",
        save=True,
    )
    
    # Auto-copy best weights to models/best.pt
    best_weights = os.path.join("runs_weed", "weed_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs("models", exist_ok=True)
        shutil.copy2(best_weights, os.path.join("models", "best.pt"))
        print(f"SUCCESS: Best trained weights deployed to models/best.pt!")
    
    print("TRAIN DONE")
