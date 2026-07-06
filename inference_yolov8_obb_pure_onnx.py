"""
inference_yolov8_obb_pure_onnx.py
Reimplementation of inference_yolov8_obb using ONLY onnxruntime, numpy and opencv.
No ultralytics / torch dependency.
"""
import os
import math
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ─── 1. Letterbox (same as ultralytics LetterBox) ─────────────────────────────
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad to new_shape keeping aspect ratio."""
    h0, w0 = img.shape[:2]
    target_w, target_h = new_shape
    r = min(target_w / w0, target_h / h0)
    nw = int(round(w0 * r))
    nh = int(round(h0 * r))
    dw = (target_w - nw) / 2.0
    dh = (target_h - nh) / 2.0
    if (w0, h0) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

# ─── 2. xywhr → 4 corner points (same math as ultralytics xywhr2xyxyxyxy) ────
def xywhr2xyxyxyxy(rboxes):
    """rboxes: (N, 5) – [cx, cy, w, h, angle_rad]  →  (N, 4, 2)"""
    ctr   = rboxes[:, :2]
    w, h, angle = rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    cos_a = np.cos(angle);  sin_a = np.sin(angle)
    vec1 = np.stack([ w / 2 * cos_a,  w / 2 * sin_a], axis=1)
    vec2 = np.stack([-h / 2 * sin_a,  h / 2 * cos_a], axis=1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=1)        # (N, 4, 2)

# ─── 3. Pure-ONNX post-processing (mirrors ultralytics OBBPredictor) ──────────
def postprocess(pred_raw, img_shape_hw, orig_shape_hw, conf_thr=0.25, iou_thr=0.45):
    """
    pred_raw : (1, 13, 8400)  – raw model output
    img_shape_hw  : (H, W) of the padded/resized tensor fed to the model (e.g. 640×640)
    orig_shape_hw : (H, W) of the ORIGINAL image (before letterbox)
    Returns list of (poly_4x2, cls_id_int, conf_float) in ORIGINAL image space.
    """
    preds = pred_raw[0].T                    # (8400, 13)
    num_classes = preds.shape[1] - 5         # 13 - 4(xywh) - 1(angle) = 8
    boxes  = preds[:, :4]                    # cx, cy, w, h  (in letterboxed space)
    scores = preds[:, 4:4 + num_classes]
    angles = preds[:, -1]

    max_scores = np.max(scores, axis=1)
    class_ids  = np.argmax(scores, axis=1)

    # loose pre-filter before NMS
    keep = max_scores >= 0.01
    boxes      = boxes[keep]
    max_scores = max_scores[keep]
    class_ids  = class_ids[keep]
    angles     = angles[keep]

    if len(boxes) == 0:
        return []

    # NMS on rotated boxes
    rotated = [
        ((float(cx), float(cy)), (float(w), float(h)), math.degrees(float(a)))
        for (cx, cy, w, h), a in zip(boxes, angles)
    ]
    idxs = cv2.dnn.NMSBoxesRotated(rotated, max_scores.tolist(), 0.01, iou_thr)

    results = []
    if len(idxs) > 0:
        for idx in idxs.flatten():
            conf = float(max_scores[idx])
            if conf < conf_thr:
                continue

            # ── rescale box center+size from padded space → original space ──
            # mirrors ultralytics ops.scale_boxes(img.shape[2:], rboxes[:,:4], orig_img.shape, xywh=True)
            ih, iw = img_shape_hw
            oh, ow = orig_shape_hw
            gain = min(iw / ow, ih / oh)
            pad_x = (iw - round(ow * gain)) / 2.0
            pad_y = (ih - round(oh * gain)) / 2.0
            # integer pads (matching what letterbox actually applied)
            pad_x = int(round(pad_x - 0.1))
            pad_y = int(round(pad_y - 0.1))

            cx, cy, w, h = boxes[idx]
            cx = (cx - pad_x) / gain
            cy = (cy - pad_y) / gain
            w  = w  / gain
            h  = h  / gain

            rbox = np.array([[cx, cy, w, h, float(angles[idx])]])
            poly = xywhr2xyxyxyxy(rbox)[0]        # (4, 2) in original image space

            results.append((poly, int(class_ids[idx]), conf))

    return results

# ─── 4. Main inference function (drop-in for inference_yolov8_obb) ────────────
def inference_yolov8_obb_onnx(
    model_path,
    image_path,
    class_names=None,
    conf_threshold=0.25,
    input_size=(640, 640),   # must match what the model was trained/exported with
    save_path=None,
    show=True,
):
    # Load session once
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Pre-process
    padded, _, _ = letterbox(image, new_shape=input_size)
    inp = padded[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, /255
    inp = inp.transpose(2, 0, 1)[None]                     # HWC→NCHW

    # Inference
    raw = session.run(None, {input_name: inp})[0]   # (1, 13, 8400)

    # Post-process
    detections = postprocess(
        raw,
        img_shape_hw=input_size,
        orig_shape_hw=(orig_h, orig_w),
        conf_thr=conf_threshold,
    )

    if class_names is None:
        class_names = []

    if detections:
        print(f"Found {len(detections)} OBB detections")
        for poly, cls_id, conf in detections:
            pts = np.round(poly).astype(np.int32).reshape((-1, 2))
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            label = f"{cls_name}: {conf:.2f}"
            cv2.polylines(image_rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            centroid = tuple(np.mean(pts, axis=0).astype(int))
            cv2.putText(
                image_rgb, label,
                (max(5, centroid[0]), max(20, centroid[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (238, 75, 43), 2, cv2.LINE_AA,
            )
    else:
        print("No detections found")

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"Detections: {os.path.basename(image_path)}")
        plt.show()

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {save_path}")

    return image_rgb


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_path = hf_hub_download(repo_id="Nicias/det_om", filename="best.onnx")
    img_path = r"D:\psi_work\ai_project\momo_ai\momo_ai_dev_space\model_dev\5.Type\CNN\data\SEMO\raw_image_OM_00018.png"

    class_names = [
        "STRUCTURE", "NOM", "TITRE", "INDICE", "PROVENANCEE",
        "DATE_DEPART", "DATE_RETOUR", "SIGNATAIRE"
    ]

    detections = inference_yolov8_obb_onnx(
        model_path=model_path,
        image_path=img_path,
        class_names=class_names,
        conf_threshold=0.25,
        show=False,
    )

    print("\nDetections:")
    # Now compare with ultralytics
    from ultralytics import YOLO
    import torch
    yolo = YOLO(model_path, task='obb')
    results_yolo = yolo.predict(source=img_path, imgsz=640, rect=False, conf=0.25, save=False, verbose=False)
    r_yolo = results_yolo[0]
    yolo_boxes = r_yolo.obb.xyxyxyxy.cpu().numpy()
    yolo_confs = r_yolo.obb.conf.cpu().numpy()
    yolo_cls   = r_yolo.obb.cls.cpu().numpy().astype(int)

    # We need a session too for pure ONNX
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    image = cv2.imread(img_path)
    padded, _, _ = letterbox(image, new_shape=(640, 640))
    inp = padded[:, :, ::-1].astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]
    raw = session.run(None, {input_name: inp})[0]
    onnx_dets = postprocess(raw, img_shape_hw=(640,640), orig_shape_hw=image.shape[:2], conf_thr=0.25)

    print(f"\nUltralytics: {len(yolo_boxes)} boxes | Pure ONNX: {len(onnx_dets)} boxes")
    print("\n--- Comparison ---")
    print(f"{'':5} {'CLS':5} {'CONF YOLO':12} {'CONF ONNX':12} {'MAX_DELTA_PX':12}")

    for i, (ybox, ycls, yconf) in enumerate(zip(yolo_boxes, yolo_cls, yolo_confs)):
        # find matching onnx detection by class and closest conf
        match = None
        best_dist = 9999
        for (opoly, ocls, oconf) in onnx_dets:
            if ocls == int(ycls):
                dist = abs(oconf - float(yconf))
                if dist < best_dist:
                    best_dist = dist
                    match = (opoly, ocls, oconf)
        if match is not None:
            opoly, ocls, oconf = match
            max_px = np.max(np.abs(ybox - opoly))
            print(f"Box {i}: cls={ycls:2d}, yolo_conf={yconf:.4f}, onnx_conf={oconf:.4f}, max_delta_px={max_px:.2f}")
        else:
            print(f"Box {i}: cls={ycls:2d}, yolo_conf={yconf:.4f}, onnx_conf=NO MATCH")
