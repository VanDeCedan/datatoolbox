from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import numpy as np

model_path = hf_hub_download(repo_id="Nicias/det_om", filename="best.onnx")
yolo_model = YOLO(model_path, task='obb')

img_path = r"D:\psi_work\ai_project\momo_ai\momo_ai_dev_space\model_dev\5.Type\CNN\data\SEMO\raw_image_OM_00018.png"

results_yolo = yolo_model.predict(source=img_path, imgsz=640, rect=False, conf=0.25, save=False, verbose=False)
r_yolo = results_yolo[0]

if hasattr(r_yolo, "obb") and r_yolo.obb is not None:
    obb = r_yolo.obb
    # Print the raw tensor before any scaling back if possible, or print the scaling parameters.
    # Let's print:
    # 1. obb.xyxyxyxy (original space)
    # 2. obb.xywhr (original space)
    # 3. obb.xyxy (original space)
    # 4. obb.orig_shape
    print("obb.orig_shape:", obb.orig_shape)
    print("obb.xywhr[0]:", obb.xywhr[0].cpu().numpy().tolist())
    print("obb.xyxyxyxy[0]:", obb.xyxyxyxy[0].cpu().numpy().tolist())
    
    # Let's inspect the math inside Ultralytics:
    # In YOLOv8, how is xywhr scaled?
    # Let's print predictor.scale_boxes or similar info
    predictor = yolo_model.predictor
    # The scaling is done inside the Post-predictor
    # Let's print the actual ratio and pad used by the predictor:
    # predictor.scale_boxes? Or where is it?
    # Actually, let's print the results list properties
    print("r_yolo.path:", r_yolo.path)
    print("r_yolo.names:", r_yolo.names)
