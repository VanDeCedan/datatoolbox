"""
deskewer.py
===========
Wrapper around the ResNet18 rotation-classification model trained in
2.Deskew/deskew_model.ipynb.

The model predicts the skew angle (one of 13 classes) from an image, then
the correction is applied by rotating the image by -predicted_angle degrees.

Usage
-----
    deskewer = CardDeskewer()
    deskewed_img, angle, confidence = deskewer.deskew(pil_image)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------------------------------- #
# Constants — must match the training configuration in deskew_model.ipynb
# --------------------------------------------------------------------------- #
DESKEW_WEIGHTS_PATH = Path(__file__).parent / "weights" / "best_deskew_model.onnx"
HF_REPO_ID = "Nicias/card_deskewer"
HF_FILENAME = "best_deskew_model.onnx"

# 14 angle classes used during training
DESKEW_ANGLES = [0, 15, 30, 60, 90, 120, 150, 180, 195, 210, 240, 270, 300, 330]

# ImageNet normalisation (same as training transforms)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def transform_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, convert to tensor, normalize."""
    img = pil_img.resize((224, 224), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = (img_arr - IMG_MEAN) / IMG_STD
    # HWC to CHW
    img_arr = img_arr.transpose((2, 0, 1))
    # Add batch dimension
    return np.expand_dims(img_arr, axis=0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #
class CardDeskewer:
    """
    Loads the deskew ONNX model once and exposes a simple deskew() interface.
    """

    def __init__(
        self,
        weights: Optional[Path] = None,
        angles: list[int] | None = None,
    ):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        
        self.angles = angles if angles is not None else DESKEW_ANGLES
        self._class_to_angle = {idx: a for idx, a in enumerate(self.angles)}
        
        if weights is None:
            weights = DESKEW_WEIGHTS_PATH
            if not weights.exists():
                try:
                    # Download from Hugging Face and cache locally
                    downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
                    weights = Path(downloaded_path)
                except Exception as e:
                    print(f"Warning: Failed to download model from Hugging Face: {e}")

        self.session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ---------------------------------------------------------------------- #
    def deskew(
        self,
        image: Image.Image,
        crop_borders: bool = True,
    ) -> tuple[Image.Image, int, float]:
        img_rgb = image.convert("RGB")
        img_array = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        
        # Preprocessing: resize to 640x640 and grayscale -> BGR
        resized_img = cv2.resize(img_array, (640, 640), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        pred_img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        pred_img_pil = Image.fromarray(cv2.cvtColor(pred_img_cv, cv2.COLOR_BGR2RGB))
        tensor = transform_image(pred_img_pil)

        # Inference
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]
        probs = softmax(logits)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        predicted_angle = self._class_to_angle[pred_class]

        # Apply correction: rotate by the negative of the predicted angle
        deskewed = img_rgb.rotate(-predicted_angle, expand=True)

        if crop_borders:
            gray = deskewed.convert("L")
            bbox = gray.getbbox()
            if bbox:
                deskewed = deskewed.crop(bbox)

        return deskewed, predicted_angle, confidence

    # ---------------------------------------------------------------------- #
    def top_k_predictions(
        self,
        image: Image.Image,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Return the top-k (angle, probability) pairs for diagnostics.
        """
        img_rgb = image.convert("RGB")
        img_array = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        
        resized_img = cv2.resize(img_array, (640, 640), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        pred_img_cv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        pred_img_pil = Image.fromarray(cv2.cvtColor(pred_img_cv, cv2.COLOR_BGR2RGB))
        tensor = transform_image(pred_img_pil)

        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]
        probs = softmax(logits)[0]
        
        num_classes = len(self.angles)
        top_k = min(k, num_classes)
        topk_idx = np.argsort(probs)[::-1][:top_k]
        topk_probs = probs[topk_idx]
        
        return [
            (self._class_to_angle[int(idx)], float(p))
            for idx, p in zip(topk_idx, topk_probs)
        ]
