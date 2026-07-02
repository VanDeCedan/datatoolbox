"""
recognizer.py
PARSeq-based OCR logic using custom HF checkpoint (nom and date) with ONNX Runtime.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

def get_parseq_transform(img_size):
    """
    Returns a function that takes a PIL image and returns a numpy array.
    Exact preprocessing used by PARSeq training / inference (read.py):
    Resize with BICUBIC → ToTensor → Normalize(0.5, 0.5)
    """
    def transform(image: Image.Image) -> np.ndarray:
        w, h = img_size[1], img_size[0]
        # BICUBIC resize
        img = image.convert("RGB").resize((w, h), Image.BICUBIC)
        # ToTensor: scale to 0-1 and CHW
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        # Normalize: (x - 0.5) / 0.5 = x * 2 - 1
        img_arr = (img_arr - 0.5) / 0.5
        # Add batch dim
        return np.expand_dims(img_arr, axis=0)
    return transform

def parseq_decode(logits: np.ndarray, charset: str) -> tuple[str, float]:
    """
    Decode PARSeq logits into (text, mean_confidence).
    logits shape: (1, seq_len, num_classes)
    Dans PARSeq, le dictionnaire de tokens est construit ainsi : [EOS] + charset + [BOS] + [PAD]
    """
    eos_id = 0
    itos = ['[E]'] + list(charset) + ['[B]', '[P]']
    
    # Softmax over last dimension for confidence
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = e_x / e_x.sum(axis=-1, keepdims=True)
    
    # Argmax
    preds = np.argmax(probs, axis=-1)  # (1, seq_len)
    
    # We only take the first sequence in the batch
    pred = preds[0]
    prob = probs[0]
    
    try:
        eos_idx = list(pred).index(eos_id)
        pred = pred[:eos_idx]
    except ValueError:
        pass # Pas de EOS trouvé
        
    text = ""
    confidences = []
    
    for i, idx in enumerate(pred):
        if idx < len(itos) and idx != 0 and itos[idx] not in ('[B]', '[P]'):
            text += itos[idx]
            confidences.append(prob[i, idx])
            
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, mean_conf


# ---------------------------------------------------------------------------
# Custom PARSeq recognizer — fine-tuned on names (Nicias/ocr_nom)
# ---------------------------------------------------------------------------

HF_REPO_ID  = "Nicias/ocr_nom"
HF_FILENAME = "last_ar_patched.onnx"
CHARSET_NOM = "ABCDEFGHIJKLMNOPQRSTUVWXYZépouse.-' "

class ParseqNomRecognizer:
    """
    PARSeq recognizer fine-tuned on name recognition using ONNX Runtime.
    """

    def __init__(self):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        
        try:
            ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        except Exception as e:
            print(f"Warning: Failed to download model from Hugging Face: {e}")
            raise e

        self.session = ort.InferenceSession(ckpt_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # PARSeq typically uses 32x128 for images
        # We can infer it from the ONNX input shape if available, but 32x128 is standard
        # input shape is usually [batch, channels, height, width]
        input_shape = self.session.get_inputs()[0].shape
        # Example shape: ['batch_size', 3, 32, 128]
        h = input_shape[2] if isinstance(input_shape[2], int) else 32
        w = input_shape[3] if isinstance(input_shape[3], int) else 128
        
        self.img_size = (h, w)
        self.transform = get_parseq_transform(self.img_size)

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        """Takes a PIL image, resizes it, and returns the predicted text and confidence."""
        tensor = self.transform(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]
        return parseq_decode(logits, CHARSET_NOM)

    def get_resized_image(self, image: Image.Image) -> Image.Image:
        """Returns the image resized to PARSeq expected dimensions (W, H)."""
        w, h = self.img_size[1], self.img_size[0]
        return image.convert("RGB").resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Custom PARSeq recognizer — fine-tuned on expiration dates (Nicias/ocr_carte_date_expiration)
# ---------------------------------------------------------------------------

HF_REPO_ID_DATE  = "Nicias/ocr_carte_date_expiration"
HF_FILENAME_DATE = "last_ar_patched.onnx"
CHARSET_DATE = "0123456789/PBSD "

class ParseqDateRecognizer:
    """
    PARSeq recognizer fine-tuned on expiration date recognition using ONNX Runtime.
    """

    def __init__(self):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        
        try:
            ckpt_path = hf_hub_download(repo_id=HF_REPO_ID_DATE, filename=HF_FILENAME_DATE)
        except Exception as e:
            print(f"Warning: Failed to download model from Hugging Face: {e}")
            raise e

        self.session = ort.InferenceSession(ckpt_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        h = input_shape[2] if isinstance(input_shape[2], int) else 32
        w = input_shape[3] if isinstance(input_shape[3], int) else 128
        
        self.img_size = (h, w)
        self.transform = get_parseq_transform(self.img_size)

    def recognize(self, image: Image.Image) -> tuple[str, float]:
        """Takes a PIL image, resizes it, and returns the predicted text and confidence."""
        tensor = self.transform(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = outputs[0]
        return parseq_decode(logits, CHARSET_DATE)

    def get_resized_image(self, image: Image.Image) -> Image.Image:
        """Returns the image resized to PARSeq expected dimensions (W, H)."""
        w, h = self.img_size[1], self.img_size[0]
        return image.convert("RGB").resize((w, h), Image.BICUBIC)
