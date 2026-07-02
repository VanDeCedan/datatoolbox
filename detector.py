"""
detector.py
YOLO-based card detection and cropping logic using ONNX Runtime.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

def letterbox_image_cv(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)) -> tuple[np.ndarray, float, tuple[float, float]]:
    shape = img.shape[:2]
    target_w, target_h = new_shape
    r = min(target_w / shape[1], target_h / shape[0])
    new_unpad_w = int(round(shape[1] * r))
    new_unpad_h = int(round(shape[0] * r))
    dw = (target_w - new_unpad_w) / 2.0
    dh = (target_h - new_unpad_h) / 2.0
    if (shape[1], shape[0]) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WEIGHTS_PATH = Path(__file__).parent / "weights" / "classifier.onnx"
HF_REPO_ID = "Nicias/card_cropper"
HF_FILENAME = "best.onnx"
CLASS_NAMES = ["CEDEAO", "CIP", "CNI", "PASSEPORT"]
CONFIDENCE_THRESHOLD = 0.80

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One detected (and cropped) card."""
    class_name: str
    confidence: float
    cropped_image: Image.Image
    source_file: str          # original filename (no path)
    page: Optional[int] = None  # PDF page number (1-based), None for plain images
    det_index: int = 0        # index among detections in the same source/page
    class_id: int = -1
    box_xywhn: Optional[tuple[float, float, float, float]] = None


@dataclass
class SkippedItem:
    """An image / PDF page where every detection was below the threshold."""
    source_file: str
    page: Optional[int] = None
    reason: str = "no_detection"   # "no_detection" | "low_confidence"
    confidences: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Detector class
# --------------------------------------------------------------------------- #
class CardDetector:
    """Wraps a YOLO ONNX model and exposes a simple detect() interface."""

    def __init__(self, weights: Optional[Path] = None, conf_threshold: float = CONFIDENCE_THRESHOLD):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        
        self.conf_threshold = conf_threshold
        
        if weights is None:
            weights = WEIGHTS_PATH
            if not weights.exists():
                try:
                    downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
                    weights = Path(downloaded_path)
                except Exception as e:
                    print(f"Warning: Failed to download model from Hugging Face: {e}")

        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ---------------------------------------------------------------------- #
    def detect(
        self,
        image: Image.Image,
        source_file: str,
        page: Optional[int] = None,
    ) -> tuple[list[Detection], list[SkippedItem]]:
        img_array = np.array(image)
        # Convert RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        pred_size = (640, 640)
        padded_img, scale, (pad_w, pad_h) = letterbox_image_cv(img_bgr, new_shape=pred_size)
        
        # Convert to RGB, normalize, CHW
        pred_img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        input_tensor = pred_img_rgb.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # NCHW

        # Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        preds = outputs[0]  # Shape: (1, 4 + num_classes, 8400)
        preds = np.squeeze(preds, axis=0)  # Shape: (4 + num_classes, 8400)
        preds = preds.transpose()  # Shape: (8400, 4 + num_classes)

        boxes = []
        confidences = []
        class_ids = []
        
        all_confidences = []

        for i in range(preds.shape[0]):
            box = preds[i, :4]
            scores = preds[i, 4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            all_confidences.append(float(confidence))
            
            # Use a slightly lower threshold for NMS
            if confidence >= 0.01:
                cx, cy, w, h = box
                x_min = cx - w / 2
                y_min = cy - h / 2
                boxes.append([x_min, y_min, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        detections: list[Detection] = []
        skipped: list[SkippedItem] = []

        if not boxes:
            skipped.append(SkippedItem(
                source_file=source_file,
                page=page,
                reason="no_detection",
            ))
            return detections, skipped

        # Run NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.01, nms_threshold=0.45)
        
        det_index = 0
        low_conf = []
        
        if len(indices) > 0:
            for idx in indices.flatten():
                conf = confidences[idx]
                if conf < self.conf_threshold:
                    low_conf.append(conf)
                    continue

                cls_id = class_ids[idx]
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                x_min, y_min, w, h = boxes[idx]
                x1_p, y1_p, x2_p, y2_p = x_min, y_min, x_min + w, y_min + h

                # Map back to original image coordinates
                x1 = (x1_p - pad_w) / scale
                y1 = (y1_p - pad_h) / scale
                x2 = (x2_p - pad_w) / scale
                y2 = (y2_p - pad_h) / scale

                x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                x1_int = max(0, x1_int)
                y1_int = max(0, y1_int)
                x2_int = min(image.width, x2_int)
                y2_int = min(image.height, y2_int)
                
                if x2_int <= x1_int or y2_int <= y1_int:
                    continue
                crop = image.crop((x1_int, y1_int, x2_int, y2_int))

                w_norm = (x2 - x1) / image.width
                h_norm = (y2 - y1) / image.height
                x_center_norm = x1 / image.width + w_norm / 2.0
                y_center_norm = y1 / image.height + h_norm / 2.0
                box_xywhn = (x_center_norm, y_center_norm, w_norm, h_norm)

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    cropped_image=crop,
                    source_file=source_file,
                    page=page,
                    det_index=det_index,
                    class_id=cls_id,
                    box_xywhn=box_xywhn,
                ))
                det_index += 1

        if det_index == 0 and low_conf:
            skipped.append(SkippedItem(
                source_file=source_file,
                page=page,
                reason="low_confidence",
                confidences=low_conf,
            ))
        elif det_index == 0:
            skipped.append(SkippedItem(
                source_file=source_file,
                page=page,
                reason="no_detection",
            ))

        return detections, skipped
