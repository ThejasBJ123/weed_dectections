"""
AI-Powered Computer Vision Weed Detection Module using Ultralytics YOLOv8.
Handles model loading, inference, bounding box rendering, density estimation,
and result exporting.
"""
import os
import time
import datetime
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from PIL import Image

# Ensure Ultralytics settings avoid restricted paths on Windows sandbox
os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(os.getcwd(), '.ultralytics'))

_MODEL_INSTANCE = None
_MODEL_PATH_LOADED = None


def find_model_weights() -> str:
    """
    Locates trained model weights file in the repository.
    """
    candidates = [
        os.path.join(os.getcwd(), 'models', 'best.pt'),
        os.path.join(os.getcwd(), 'weights', 'best.pt'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'best.pt')
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_yolo_model(weights_path: str = None):
    """
    Loads and caches the YOLO object detection model.
    """
    global _MODEL_INSTANCE, _MODEL_PATH_LOADED

    if weights_path is None:
        weights_path = find_model_weights()

    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Trained YOLO weights ('best.pt') not found in 'models/' or 'weights/' directory. "
            "Please check models/best.pt."
        )

    if _MODEL_INSTANCE is None or _MODEL_PATH_LOADED != weights_path:
        from ultralytics import YOLO
        _MODEL_INSTANCE = YOLO(weights_path)
        _MODEL_PATH_LOADED = weights_path

    return _MODEL_INSTANCE


def run_weed_detection(
    image_input,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    quadrat_area_m2: float = 1.0,
    model_path: str = None
) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Runs weed detection on an input image.
    
    Args:
        image_input: PIL Image, NumPy array, or path string
        conf_threshold: minimum confidence score (0.0 to 1.0)
        iou_threshold: NMS IoU threshold (0.0 to 1.0)
        quadrat_area_m2: Ground coverage area represented in m² (default 1.0m²)
        model_path: Optional custom model path
        
    Returns:
        tuple: (annotated_pil_image, detections_list, summary_metrics)
    """
    model = get_yolo_model(model_path)

    # Ensure input is a clean PIL Image
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('RGB')
    else:
        pil_img = Image.fromarray(image_input).convert('RGB')

    orig_w, orig_h = pil_img.size

    # Run inference directly with pil_img to maintain natural RGB color space
    start_time = time.time()
    results = model.predict(
        source=pil_img,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False
    )
    inference_time_ms = round((time.time() - start_time) * 1000, 1)

    result = results[0]
    detections = []

    # Prepare image for annotation using OpenCV BGR
    img_np = np.array(pil_img)
    annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = model.names

        for idx, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
            x1, y1, x2, y2 = [int(v) for v in box]
            class_name = names.get(cls_id, f"class_{cls_id}")
            
            detections.append({
                'id': idx + 1,
                'class': class_name,
                'confidence': round(float(conf), 3),
                'box': [x1, y1, x2, y2],
                'box_width': x2 - x1,
                'box_height': y2 - y1
            })

            # Color coding: Greenish for high confidence, orange for medium
            box_color = (30, 220, 50) if conf >= 0.50 else (0, 165, 255) # BGR
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), box_color, 2)

            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                annotated_bgr,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 6, max(text_h + 6, y1)),
                box_color,
                -1
            )
            cv2.putText(
                annotated_bgr,
                label,
                (x1 + 3, max(text_h + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # Convert back to PIL
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Compute summary metrics
    total_detections = len(detections)
    avg_confidence = round(float(np.mean([d['confidence'] for d in detections])), 2) if total_detections > 0 else 0.0
    
    # Estimated weeds per m² = count / quadrat_area_m2
    est_density = round(total_detections / max(0.1, quadrat_area_m2), 1)

    # Classify severity
    if est_density < 12:
        severity = 'Low'
    elif est_density <= 29:
        severity = 'Medium'
    else:
        severity = 'High'

    summary = {
        'total_detections': total_detections,
        'avg_confidence': avg_confidence,
        'inference_time_ms': inference_time_ms,
        'quadrat_area_m2': quadrat_area_m2,
        'estimated_weed_density_m2': est_density,
        'infestation_severity': severity,
        'image_dimensions': f"{orig_w}x{orig_h}",
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return annotated_pil, detections, summary


def save_detection_artifacts(
    annotated_image: Image.Image,
    summary: Dict[str, Any],
    output_dir: str = 'outputs/annotated_images'
) -> str:
    """
    Saves the annotated image to disk and returns the saved file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"detected_weeds_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    annotated_image.save(filepath, format='JPEG', quality=95)
    return filepath
