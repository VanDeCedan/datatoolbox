"""
obb_detector.py
YOLO OBB-based text detection, cropping with deskew, and dataset generation.
Using ONNX Runtime.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import math

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
HF_REPO_ID = "Nicias/rec_bio_text_boxes"
HF_FILENAME = "last.onnx"
CONFIDENCE_THRESHOLD = 0.25

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ObbDetection:
    class_id: int
    class_name: str
    confidence: float
    poly: np.ndarray  # Shape (4, 2)
    cropped_image: Optional[Image.Image] = None
    det_index: int = 0

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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
    return img, r, (left, top)

def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE and Sharpening to improve text readability for annotation."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
    img_sharpened = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
    return img_sharpened

def xywhr2xyxyxyxy(rboxes):
    """
    Convert (cx, cy, w, h, r) to 4 corner points (x, y)
    r is angle in radians.
    Returns array of shape (N, 4, 2)
    """
    ctr = rboxes[:, :2]
    w, h, angle = rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    
    vec1 = np.stack([w / 2 * cos_val, w / 2 * sin_val], axis=1)
    vec2 = np.stack([-h / 2 * sin_val, h / 2 * cos_val], axis=1)
    
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    
    return np.stack([pt1, pt2, pt3, pt4], axis=1)

class ObbDetector:
    def __init__(self, conf_threshold: float = CONFIDENCE_THRESHOLD, repo_id: str = HF_REPO_ID, filename: str = HF_FILENAME, names: dict = None):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        
        self.conf_threshold = conf_threshold
        try:
            weights = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        except Exception as e:
            print(f"Warning: Failed to download model from Hugging Face: {e}")
            raise e

        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # Hardcoding standard names for now, or parsing from ONNX if possible
        if names is not None:
            self.names = names
        else:
            self.names = {0: "ZONE1", 1: "ZONE2", 2: "ZONE3", 3: "ZONE4"}

    def _deskew_crop(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 0 or maxHeight <= 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        return warped

    def _run_inference(self, img_bgr: np.ndarray, pred_size=(640, 640)):
        orig_h, orig_w = img_bgr.shape[:2]
        padded_img, scale, (pad_left, pad_top) = letterbox_image_cv(img_bgr, new_shape=pred_size)
        pred_img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        input_tensor = pred_img_rgb.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # NCHW

        outputs = self.session.run(None, {self.input_name: input_tensor})
        preds = outputs[0]  # Shape: (1, 4 + num_classes + 1, N_anchors)
        preds = np.squeeze(preds, axis=0).transpose()  # (N_anchors, 4 + num_classes + 1)
        
        # YOLOv8 OBB output format: cx, cy, w, h, class_scores..., angle
        num_classes = preds.shape[1] - 5
        boxes  = preds[:, :4]          # cx, cy, w, h  in padded-image space
        scores = preds[:, 4:4 + num_classes]
        angles = preds[:, -1]
        
        max_scores = np.max(scores, axis=1)
        class_ids  = np.argmax(scores, axis=1)
        
        mask = max_scores >= 0.01   # loose pre-filter before NMS
        boxes      = boxes[mask]
        max_scores = max_scores[mask]
        class_ids  = class_ids[mask]
        angles     = angles[mask]
        
        if len(boxes) == 0:
            return [], []
        
        # NMSBoxesRotated
        rotated_boxes = [
            ((float(cx), float(cy)), (float(w), float(h)), math.degrees(float(a)))
            for (cx, cy, w, h), a in zip(boxes, angles)
        ]
        indices = cv2.dnn.NMSBoxesRotated(rotated_boxes, max_scores.tolist(), 0.01, 0.45)
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                conf = float(max_scores[idx])
                if conf < self.conf_threshold:
                    continue
                
                cls_id = class_ids[idx]
                cx, cy, w, h = boxes[idx]
                angle = float(angles[idx])

                # ── scale_boxes(xywh=True): subtract pad then divide by gain ──
                # This mirrors: ops.scale_boxes(img.shape[2:], rboxes[:,:4], orig_img.shape, xywh=True)
                ih, iw = pred_size
                gain = min(iw / orig_w, ih / orig_h)
                px = int(round((iw - round(orig_w * gain)) / 2.0 - 0.1))
                py = int(round((ih - round(orig_h * gain)) / 2.0 - 0.1))
                cx = (cx - px) / gain
                cy = (cy - py) / gain
                w  = w  / gain
                h  = h  / gain

                rbox = np.array([[cx, cy, w, h, angle]])
                poly = xywhr2xyxyxyxy(rbox)[0]  # Shape (4, 2) in original image space

                results.append((poly, int(cls_id), conf))
                
        return results, (padded_img, scale, pad_left, pad_top)


    def detect_and_crop(self, image: Image.Image) -> list[ObbDetection]:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results, _ = self._run_inference(img_array)
        
        detections = []
        for i, (poly, cls_id, conf) in enumerate(results):
            warped = self._deskew_crop(img_array, poly)
            enhanced_crop = enhance_image(warped)
            crop_pil = Image.fromarray(cv2.cvtColor(enhanced_crop, cv2.COLOR_BGR2RGB))
            class_name = self.names.get(cls_id, str(cls_id))
            
            detections.append(ObbDetection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                poly=poly,
                cropped_image=crop_pil,
                det_index=i
            ))
        return detections

    def detect_and_generate_labels(self, image: Image.Image, target_size=(640, 640)) -> tuple[Image.Image, str]:
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Prepare target dataset image
        padded_img, target_scale, (target_pad_w, target_pad_h) = letterbox_image_cv(img_array, new_shape=target_size)
        
        # 2. Run inference on original image
        results, _ = self._run_inference(img_array)
        
        labels_text = ""
        for poly, cls_id, conf in results:
            # Map original image points to target dataset image
            pts = poly.copy()
            pts[:, 0] = pts[:, 0] * target_scale + target_pad_w
            pts[:, 1] = pts[:, 1] * target_scale + target_pad_h
            
            # Normalize
            pts[:, 0] /= target_size[0]
            pts[:, 1] /= target_size[1]
            
            pts_flat = pts.flatten()
            coords_str = " ".join([f"{c:.6f}" for c in pts_flat])
            labels_text += f"{cls_id} {coords_str}\n"

        padded_pil = Image.fromarray(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
        return padded_pil, labels_text
