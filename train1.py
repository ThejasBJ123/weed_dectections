from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")
    model.train(
        data="FINAL_3064/data.yaml",
        epochs=50,
        imgsz=1280,
        batch=4,
        workers=0,
        patience=10,
        project="runs_weed",
        name="weed_80",
        save=True,
    )
    print("TRAIN DONE")