"""
Test: generate labels for a few SEMO images using pure ONNX ObbDetector
and verify the output format matches what the annotator expects:
  class_id x1n y1n x2n y2n x3n y3n x4n y4n  (all values normalized [0,1])

The annotator parses: parts[0]=classId, parts[1..8]=x1y1x2y2x3y3x4y4 normalized
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
sys.path.insert(0, r"D:\psi_work\ai_project\momo_ai\DataToolBox")

from obb_detector import ObbDetector

om_names = {
    0: "STRUCTURE", 1: "NOM", 2: "TITRE", 3: "INDICE",
    4: "PROVENANCEE", 5: "DATE_DEPART", 6: "DATE_RETOUR", 7: "SIGNATAIRE"
}

detector = ObbDetector(
    conf_threshold=0.25,
    repo_id="Nicias/det_om",
    filename="best.onnx",
    names=om_names
)

img_dir = Path(r"D:\psi_work\ai_project\momo_ai\momo_ai_dev_space\model_dev\5.Type\CNN\data\SEMO")
out_dir = Path(r"D:\psi_work\ai_project\momo_ai\DataToolBox\test_labels_output")
out_dir.mkdir(exist_ok=True)

test_images = sorted(img_dir.glob("*.png"))[:5]

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    
    results, _ = detector._run_inference(img_bgr, pred_size=(640, 640))
    
    label_lines = []
    for poly, cls_id, conf in results:
        pts = poly.copy()
        # normalize by image dimensions
        pts[:, 0] /= w
        pts[:, 1] /= h
        flat = pts.flatten()
        coords_str = " ".join([f"{c:.6f}" for c in flat])
        label_lines.append(f"{cls_id} {coords_str}")
    
    label_text = "\n".join(label_lines)
    out_path = out_dir / img_path.with_suffix(".txt").name
    out_path.write_text(label_text, encoding="utf-8")
    
    print(f"\n{img_path.name} ({w}x{h}) -> {len(results)} detections:")
    for i, line in enumerate(label_lines):
        parts = line.split()
        cls_id = parts[0]
        vals = [float(v) for v in parts[1:]]
        all_in_range = all(0.0 <= v <= 1.0 for v in vals)
        print(f"  Box {i}: class={om_names.get(int(cls_id), cls_id)}, "
              f"in_range=[0,1]={all_in_range}, "
              f"x1n={vals[0]:.4f}, y1n={vals[1]:.4f}")
    
    print(f"  Label saved to: {out_path}")

print("\n\nSample label file content (first image):")
first_label = list(out_dir.glob("*.txt"))[0]
print(first_label.read_text(encoding="utf-8"))
