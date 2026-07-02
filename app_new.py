"""
app.py — Data Toolbox App (Gradio version)
==========================================
Three modes via tabs:
  1. Classify        — YOLO detects and classifies cards. Options to crop and deskew.
  2. Only Deskew     — Upload already-cropped cards, just deskew them.
  3. Group by Batches — Split any files into batch folders inside a ZIP.
"""

from __future__ import annotations

import os
import sys

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

# Configuration de l'encodage UTF-8 pour Windows afin d'éviter les plantages d'écriture d'émojis
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

# pyrefly: ignore [missing-import]
import gradio as gr
import pandas as pd
from PIL import Image

from detector import CardDetector, Detection, SkippedItem, CLASS_NAMES
from deskewer import CardDeskewer
from obb_detector import ObbDetector, ObbDetection
from recognizer import ParseqRecognizer, ParseqNomRecognizer
from pdf_utils import pdf_to_images, count_pdf_pages

# ---------------------------------------------------------------------------
# Model loaders (lazy, cached at module level)
# ---------------------------------------------------------------------------
_detector: Optional[CardDetector] = None
_deskewer: Optional[CardDeskewer] = None
_obb_detector: Optional[ObbDetector] = None
_parseq: Optional[ParseqRecognizer] = None
_parseq_nom: Optional[ParseqNomRecognizer] = None

def get_detector(conf_threshold: float = 0.80) -> CardDetector:
    global _detector
    if _detector is None:
        _detector = CardDetector()
    _detector.conf_threshold = conf_threshold
    return _detector


def get_deskewer() -> CardDeskewer:
    global _deskewer
    if _deskewer is None:
        _deskewer = CardDeskewer()
    return _deskewer


def get_obb_detector(conf_threshold: float = 0.25) -> ObbDetector:
    global _obb_detector
    if _obb_detector is None:
        _obb_detector = ObbDetector()
    _obb_detector.conf_threshold = conf_threshold
    return _obb_detector


def get_parseq_recognizer() -> ParseqRecognizer:
    global _parseq
    if _parseq is None:
        _parseq = ParseqRecognizer()
    return _parseq


def get_parseq_nom_recognizer() -> ParseqNomRecognizer:
    global _parseq_nom
    if _parseq_nom is None:
        _parseq_nom = ParseqNomRecognizer()
    return _parseq_nom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_path(uf) -> str:
    """Normalize Gradio file objects to a plain string path.
    - Gradio 4: object with .name
    - Gradio 5: FileData object with .path, or plain str
    """
    if isinstance(uf, str):
        return uf
    if hasattr(uf, 'path') and uf.path:
        return uf.path
    if hasattr(uf, 'name'):
        return uf.name
    return str(uf)


def save_crop_to_disk(det: Detection, crop_dir: Path, original_image=None) -> str:
    class_dir = crop_dir / det.class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(det.source_file).stem
    page_part = f"_p{det.page:03d}" if det.page is not None else ""
    fname = f"{stem}{page_part}_det{det.det_index:02d}.jpg"
    out_path = class_dir / fname
    img_to_save = original_image if original_image is not None else det.cropped_image
    img_to_save.save(str(out_path), "JPEG", quality=95)
    return str(out_path)


def save_deskewed_to_disk(img: Image.Image, dest_dir: Path, stem: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{stem}_deskewed.jpg"
    img.save(str(out_path), "JPEG", quality=95)
    return str(out_path)


def build_zip_to_file(root_dir: Path, zip_path: Path) -> None:
    """Build ZIP from all files under root_dir directly to a file on disk, preserving relative paths."""
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(root_dir.rglob("*")):
            if f.is_file():
                arcname = f.relative_to(root_dir).as_posix()
                zf.write(f, arcname=arcname)


# ---------------------------------------------------------------------------
# Tab 1 — Classify
# ---------------------------------------------------------------------------
def run_classify_mode(
    uploaded_files,
    conf_class_pct: int,
    mode: str,
) -> tuple[str, str | None, object | None]:
    if not uploaded_files:
        yield "⚠️ Aucun fichier uploadé.", None, None
        return

    conf_class = conf_class_pct / 100.0
    logs = []

    def log(msg: str):
        logs.append(msg)

    log("🔮 Chargement du modèle de classification…")
    detector = get_detector(conf_class)

    tmp_root = tempfile.mkdtemp()
    all_records = []
    all_skipped = []

    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "output"
        pdf_dir = tmp_path / "pdfs"
        out_dir.mkdir()
        pdf_dir.mkdir()

        if mode == "yolo_annotation":
            images_dir = out_dir / "images"
            labels_dir = out_dir / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

        BATCH_SIZE = 500
        for idx, uf in enumerate(uploaded_files):
            if idx % BATCH_SIZE == 0:
                log(f"🔄 Traitement du lot {idx//BATCH_SIZE + 1}/{(len(uploaded_files) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            fpath = _get_path(uf)
            fname = Path(fpath).name
            is_pdf = fname.lower().endswith(".pdf")
            stem = Path(fname).stem

            if is_pdf:
                log(f"📄 Traitement PDF : {fname}")
                pdf_tmp = pdf_dir / fname
                shutil.copy(fpath, pdf_tmp)
                try:
                    n_pages = count_pdf_pages(pdf_tmp)
                    for page_num, pil_img in pdf_to_images(pdf_tmp):
                        log(f"  📃 Page {page_num}/{n_pages}…")
                        dets, skipped = detector.detect(pil_img, source_file=fname, page=page_num)
                        
                        if mode == "yolo_annotation":
                            page_stem = f"{stem}_p{page_num:03d}"
                            pil_img.save(str(images_dir / f"{page_stem}.jpg"), "JPEG", quality=95)
                            labels_text = ""
                            for det in dets:
                                if det.box_xywhn is not None and det.class_id != -1:
                                    x, y, w, h = det.box_xywhn
                                    labels_text += f"{det.class_id} {x} {y} {w} {h}" + "\n"
                                all_records.append({"Fichier": fname, "Page": page_num, "Classe": det.class_name, "Conf. Det.": f"{det.confidence:.2%}"})
                                log(f"    ✅ {det.class_name} ({det.confidence:.1%})")
                            with open(labels_dir / f"{page_stem}.txt", "w", encoding="utf-8") as f:
                                f.write(labels_text)
                            for s in skipped:
                                all_skipped.append({"Fichier": fname, "Page": page_num, "Raison": s.reason})
                        else:
                            for det in dets[:1]:
                                class_dir = out_dir / det.class_name
                                class_dir.mkdir(parents=True, exist_ok=True)
                                page_part = f"_p{page_num:03d}"
                                
                                if mode == "classification_only":
                                    out_path = class_dir / f"{stem}{page_part}.jpg"
                                    pil_img.save(str(out_path), "JPEG", quality=95)
                                else:
                                    out_path = class_dir / f"{stem}{page_part}_det{det.det_index:02d}.jpg"
                                    det.cropped_image.save(str(out_path), "JPEG", quality=95)
                                
                                all_records.append({
                                    "Fichier": fname, "Page": page_num,
                                    "Classe": det.class_name, "Conf. Det.": f"{det.confidence:.2%}",
                                })
                                log(f"    ✅ {det.class_name} ({det.confidence:.1%})")
                            for s in skipped:
                                all_skipped.append({"Fichier": fname, "Page": page_num, "Raison": s.reason})
                        pil_img = None
                except Exception as exc:
                    log(f"  ❌ Erreur PDF : {exc}")
                    all_skipped.append({"Fichier": fname, "Page": "—", "Raison": "error"})
            else:
                log(f"🖼️ Traitement image : {fname}")
                try:
                    pil_img = Image.open(fpath).convert("RGB")
                    dets, skipped = detector.detect(pil_img, source_file=fname, page=None)
                    
                    if mode == "yolo_annotation":
                        pil_img.save(str(images_dir / f"{stem}.jpg"), "JPEG", quality=95)
                        labels_text = ""
                        for det in dets:
                            if det.box_xywhn is not None and det.class_id != -1:
                                x, y, w, h = det.box_xywhn
                                labels_text += f"{det.class_id} {x} {y} {w} {h}" + "\n"
                            all_records.append({"Fichier": fname, "Page": "—", "Classe": det.class_name, "Conf. Det.": f"{det.confidence:.2%}"})
                            log(f"  ✅ {det.class_name} ({det.confidence:.1%})")
                        with open(labels_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                            f.write(labels_text)
                        for s in skipped:
                            all_skipped.append({"Fichier": fname, "Page": "—", "Raison": s.reason})
                    else:
                        for det in dets[:1]:
                            class_dir = out_dir / det.class_name
                            class_dir.mkdir(parents=True, exist_ok=True)
                            
                            if mode == "classification_only":
                                out_path = class_dir / f"{stem}.jpg"
                                pil_img.save(str(out_path), "JPEG", quality=95)
                            else:
                                out_path = class_dir / f"{stem}_det{det.det_index:02d}.jpg"
                                det.cropped_image.save(str(out_path), "JPEG", quality=95)
                                
                            all_records.append({
                                "Fichier": fname, "Page": "—",
                                "Classe": det.class_name, "Conf. Det.": f"{det.confidence:.2%}",
                            })
                            log(f"  ✅ {det.class_name} ({det.confidence:.1%})")
                        for s in skipped:
                            all_skipped.append({"Fichier": fname, "Page": "—", "Raison": s.reason})
                    pil_img = None
                except Exception as exc:
                    log(f"  ❌ Erreur : {exc}")
                    all_skipped.append({"Fichier": fname, "Page": "—", "Raison": "error"})
            
            if (idx + 1) % BATCH_SIZE == 0:
                import gc; gc.collect()
                yield "\n".join(logs), None, pd.DataFrame(all_records) if all_records else None

        if not all_records:
            log("⚠️ Aucun résultat détecté.")
            return "\\n".join(logs), None, pd.DataFrame(all_skipped) if all_skipped else None

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_classify.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")
        log(f"🎉 Terminé ! {len(all_records)} résultat(s) traité(s).")

        df = pd.DataFrame(all_records)
        return "\\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        yield f"❌ Erreur inattendue : {exc}", None, None
        return


# ---------------------------------------------------------------------------
# Tab 2 — Only Deskew
# ---------------------------------------------------------------------------
def run_deskew(
    uploaded_files,
    conf_deskew_pct: int,
) -> tuple[str, Optional[str], Optional[pd.DataFrame]]:
    if not uploaded_files:
        yield "⚠️ Aucun fichier uploadé.", None, None
        return

    conf_deskew = conf_deskew_pct / 100.0
    logs = []

    def log(msg: str):
        logs.append(msg)

    log("🔮 Chargement du modèle de redressement…")
    deskewer = get_deskewer()

    tmp_root = tempfile.mkdtemp()
    all_records = []

    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "deskewed"
        out_dir.mkdir()

        BATCH_SIZE = 500
        for idx, uf in enumerate(uploaded_files):
            if idx % BATCH_SIZE == 0:
                log(f"🔄 Traitement du lot {idx//BATCH_SIZE + 1}/{(len(uploaded_files) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            fpath = _get_path(uf)
            fname = Path(fpath).name
            log(f"📐 Redressement : {fname}")
            try:
                pil_img = Image.open(fpath).convert("RGB")
                d_img, d_angle, d_conf = deskewer.deskew(pil_img, crop_borders=True)
                disk_path = save_deskewed_to_disk(d_img, out_dir, Path(fname).stem)
                if d_conf >= conf_deskew:
                    log(f"  ✅ Angle : -{d_angle}° | Confiance : {d_conf:.2f}")
                else:
                    log(f"  ⚠️ Faible confiance ({d_conf:.2f}) — orientation originale conservée")
                all_records.append({
                    "Fichier": fname,
                    "Angle détecté": f"{d_angle}°",
                    "Correction": f"-{d_angle}°",
                    "Confiance": f"{d_conf:.2%}",
                    "Statut": "OK" if d_conf >= conf_deskew else "Confiance faible",
                })
                pil_img = None
            except Exception as exc:
                log(f"  ⚠️ Erreur : {exc}")
                all_records.append({
                    "Fichier": fname, "Angle détecté": "—", "Correction": "—",
                    "Confiance": "—", "Statut": "Erreur",
                })
            
            if (idx + 1) % BATCH_SIZE == 0:
                import gc; gc.collect()
                yield "\n".join(logs), None, pd.DataFrame(all_records) if all_records else None

        if not all_records:
            yield "\n".join(logs), None, None
            return

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_deskew.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")
        log(f"🎉 Terminé ! {len(all_records)} image(s) redressée(s).")

        df = pd.DataFrame(all_records)
        yield "\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        yield f"❌ Erreur inattendue : {exc}", None, None
        return


# ---------------------------------------------------------------------------
# Tab 4 — Reconnaissance (OCR) Logic
# ---------------------------------------------------------------------------
def run_ocr_mode(
    uploaded_files,
    conf_threshold_pct: int,
    model_type: str = "general",
) -> tuple[str, str | None, object | None]:
    if not uploaded_files:
        yield "⚠️ Aucun fichier uploadé.", None, None
        return

    logs = []
    def log(msg: str): logs.append(msg)

    label = "PARSeq Générale" if model_type == "general" else "PARSeq Nom"
    log(f"🔮 Chargement du modèle OCR ({label})…")
    try:
        if model_type == "general":
            recognizer = get_parseq_recognizer()
        else:
            recognizer = get_parseq_nom_recognizer()
    except Exception as e:
        yield f"❌ Erreur de chargement du modèle: {e}", None, None
        return

    tmp_root = tempfile.mkdtemp()
    all_records = []

    try:
        tmp_path = Path(tmp_root)
        csv_data = []

        BATCH_SIZE = 500
        for idx, uf in enumerate(uploaded_files):
            if idx % BATCH_SIZE == 0:
                log(f"🔄 Traitement du lot {idx//BATCH_SIZE + 1}/{(len(uploaded_files) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            fpath = _get_path(uf)
            fname = Path(fpath).name
            stem = Path(fname).stem

            log(f"🖼️ Traitement : {fname}")
            try:
                pil_img = Image.open(fpath).convert("RGB")

                # Predict
                text, conf = recognizer.recognize(pil_img)

                text_clean = text.strip() if text else ""

                if not text_clean or conf < (conf_threshold_pct / 100.0):
                    log(f"  ⚠️ Ignoré: Texte vide ou confiance trop faible ('{text_clean}', conf: {conf:.2f})")
                    continue

                log(f"  ✅ Texte: '{text_clean}' (Confiance: {conf:.2f})")

                all_records.append({"Fichier": fname, "Texte": text_clean, "Confiance": f"{conf:.2f}"})
                csv_data.append(f"{fname},{text_clean}")

            except Exception as exc:
                log(f"  ❌ Erreur : {exc}")
            
            if (idx + 1) % BATCH_SIZE == 0:
                import gc; gc.collect()
                yield "\n".join(logs), None, pd.DataFrame(all_records) if all_records else None

        if not all_records:
            log("⚠️ Aucun résultat.")
            yield "\n".join(logs), None, None
            return

        # Write CSV only
        csv_path = tmp_path / "labels.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,text\n")
            f.write("\n".join(csv_data) + "\n")

        log(f"✅ labels.csv généré — {len(all_records)} ligne(s)")

        df = pd.DataFrame(all_records)
        yield "\n".join(logs), str(csv_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        yield f"❌ Erreur inattendue : {exc}", None, None
        return


# ---------------------------------------------------------------------------
# Tab 3 — Detection de texte
# ---------------------------------------------------------------------------
def run_obb_mode(
    uploaded_files,
    conf_pct: int,
    mode: str,
    target_size_w: int = 640,
    target_size_h: int = 640,
    card_type: str = ""
) -> tuple[str, str | None, object | None]:
    if not uploaded_files:
        yield "⚠️ Aucun fichier uploadé.", None, None
        return
        
    if mode == "crop_resize" and not card_type:
        yield "⚠️ Erreur: Le type de carte est obligatoire pour lancer le découpage.", None, None
        return

    conf = conf_pct / 100.0
    logs = []
    def log(msg: str): logs.append(msg)

    log("🔮 Chargement du modèle OBB…")
    try:
        detector = get_obb_detector(conf)
    except Exception as e:
        yield f"❌ Erreur de chargement du modèle: {e}", None, None
        return

    tmp_root = tempfile.mkdtemp()
    all_records = []
    target_size = (target_size_w, target_size_h)

    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "output_obb"
        out_dir.mkdir()

        if mode == "generate_labels":
            images_dir = out_dir / "images"
            labels_dir = out_dir / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

        BATCH_SIZE = 500
        for idx, uf in enumerate(uploaded_files):
            if idx % BATCH_SIZE == 0:
                log(f"🔄 Traitement du lot {idx//BATCH_SIZE + 1}/{(len(uploaded_files) + BATCH_SIZE - 1)//BATCH_SIZE}...")
            fpath = _get_path(uf)
            fname = Path(fpath).name
            stem = Path(fname).stem
            
            log(f"🖼️ Traitement : {fname}")
            try:
                pil_img = Image.open(fpath).convert("RGB")
                
                if mode == "generate_labels":
                    padded_pil, labels_text = detector.detect_and_generate_labels(pil_img, target_size=target_size)
                    
                    img_path = images_dir / f"{stem}.jpg"
                    lbl_path = labels_dir / f"{stem}.txt"
                    
                    padded_pil.save(str(img_path), "JPEG", quality=95)
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        f.write(labels_text)
                    
                    num_labels = len(labels_text.strip().split('\\n')) if labels_text.strip() else 0
                    log(f"  ✅ Généré {num_labels} labels pour {fname}")
                    all_records.append({"Fichier": fname, "Labels": num_labels})
                    
                elif mode == "crop_resize":
                    # For crop_resize we just detect and crop, enhancement is applied in detector
                    detections = detector.detect_and_crop(pil_img)
                    
                    if len(detections) != 4:
                        log(f"  ⚠️ Ignoré: {len(detections)} zones détectées au lieu de 4.")
                        continue

                    # Sort top to bottom by Y-center
                    detections.sort(key=lambda d: d.poly[:, 1].mean())
                    
                    prefixes = []
                    if card_type == "CEDEAO":
                        prefixes = ["NOM", "PRENOM", "DATE EXPIRATION", "NUMERO CARTE"]
                    elif card_type in ["CIP", "PASSEPORT"]:
                        prefixes = ["NUMERO CARTE", "NOM", "PRENOM", "DATE EXPIRATION"]
                    elif card_type == "CNI":
                        top_2 = sorted(detections[:2], key=lambda d: d.poly[:, 0].mean())
                        bottom_2 = sorted(detections[2:], key=lambda d: d.poly[:, 0].mean())
                        detections = top_2 + bottom_2
                        prefixes = ["NOM", "PRENOM", "NUMERO CARTE", "DATE EXPIRATION"]
                    else:
                        prefixes = ["ZONE1", "ZONE2", "ZONE3", "ZONE4"]

                    log(f"  ✅ Découpage réussi ({card_type}).")
                    
                    for i, det in enumerate(detections):
                        prefix = prefixes[i]
                        prefix_dir = out_dir / prefix
                        prefix_dir.mkdir(exist_ok=True)
                        
                        det_stem = f"{prefix}_{card_type}_{stem}.jpg"
                        out_path = prefix_dir / det_stem
                        
                        if det.cropped_image:
                            det.cropped_image.save(str(out_path), "JPEG", quality=95)
                        
                        all_records.append({
                            "Fichier Original": fname,
                            "Fichier Crop": f"{prefix}/{det_stem}",
                            "Confiance": f"{det.confidence:.2%}"
                        })

            except Exception as exc:
                log(f"  ❌ Erreur : {exc}")
            
            if (idx + 1) % BATCH_SIZE == 0:
                import gc; gc.collect()
                yield "\n".join(logs), None, pd.DataFrame(all_records) if all_records else None

        if not all_records:
            log("⚠️ Aucun résultat.")
            yield "\n".join(logs), None, None
            return

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_obb.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")

        df = pd.DataFrame(all_records)
        yield "\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        yield f"❌ Erreur inattendue : {exc}", None, None
        return


# ---------------------------------------------------------------------------
# CSS / Theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0d1b2a 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 20px !important;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 16px;
    text-align: center;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

/* Tab styling */
.tab-nav {
    background: rgba(255,255,255,0.03) !important;
    border-bottom: 2px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px 12px 0 0 !important;
}
.tab-nav button {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #a78bfa !important;
    background: rgba(139,92,246,0.12) !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* Inputs / controls */
.gr-box, .gr-form {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-radius: 12px !important;
}
label, .gr-label {
    color: #c4b5fd !important;
    font-weight: 500 !important;
}

/* Upload area */
.gr-file-drop {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(139,92,246,0.4) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
.gr-file-drop:hover {
    border-color: rgba(139,92,246,0.8) !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139,92,246,0.4) !important;
}
.gr-button-secondary {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    color: #c4b5fd !important;
    font-weight: 600 !important;
    transition: all 0.15s !important;
}
.gr-button-secondary:hover {
    background: rgba(139,92,246,0.15) !important;
    transform: translateY(-2px) !important;
}

/* Sliders */
input[type='range'] {
    accent-color: #7c3aed !important;
}

/* Textbox / log area */
textarea {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.82rem !important;
}

/* Dataframe */
.gr-dataframe {
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Number input */
input[type='number'] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    padding: 8px 12px !important;
}


/* Horizontal file list */
.horizontal-files .file-preview-holder,
.horizontal-files div[role="group"],
.horizontal-files tbody,
.horizontal-files [data-testid="file-upload"] > div:nth-child(2) {
    display: flex !important;
    flex-direction: row !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    padding-bottom: 8px !important;
}
.horizontal-files .file-preview-item,
.horizontal-files tr {
    display: flex !important;
    flex-direction: column !important;
    flex: 0 0 auto !important;
    min-width: 150px !important;
    margin-right: 10px !important;
}
"""

HERO_HTML = """
<div class="hero-banner">
    <div class="hero-title">Datatoolbox - Version carte</div>
</div>
"""



# ---------------------------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------------------------
def df_update_with_count(df, base_label="📊 Résultats"):
    if df is not None:
        return gr.update(value=df, label=f"{base_label} ({len(df)} nouveaux fichiers créés)", visible=True)
    return gr.update(value=None, visible=False)

with gr.Blocks(css=CUSTOM_CSS, title="Data Toolbox — Momo AI") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

                # ── Tab 1 : Classify ────────────────────────────────────────────────
        with gr.Tab("🔍 Classify"):
            with gr.Tabs():
                with gr.Tab("Classification"):
                    gr.Markdown("### Classification seule (les images originales sont classées dans des sous-dossiers)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            c1_files = gr.File(label="Upload Images ou PDFs", file_count="multiple", file_types=["image", ".pdf"], elem_classes=["horizontal-files"])
                            c1_files.change(fn=lambda f: gr.update(label=f"Upload Images ou PDFs ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images ou PDFs"), inputs=c1_files, outputs=c1_files)
                        with gr.Column(scale=1):
                            c1_conf = gr.Slider(minimum=40, maximum=100, value=80, step=5, label="Seuil de confiance (%)")
                    c1_btn = gr.Button("🚀 Lancer Classification", variant="primary")
                    with gr.Row():
                        c1_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        c1_dl = gr.File(label="⬇️ Télécharger ZIP", visible=False)
                        c1_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_c1(files, conf):
                        for logs, zip_p, df in run_classify_mode(files, conf, "classification_only"):
                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
                    c1_btn.click(fn_c1, inputs=[c1_files, c1_conf], outputs=[c1_log, c1_dl, c1_df])

                with gr.Tab("Classifier et découper"):
                    gr.Markdown("### Détection, recadrage et classification (les cartes recadrées sont classées)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            c2_files = gr.File(label="Upload Images ou PDFs", file_count="multiple", file_types=["image", ".pdf"], elem_classes=["horizontal-files"])
                            c2_files.change(fn=lambda f: gr.update(label=f"Upload Images ou PDFs ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images ou PDFs"), inputs=c2_files, outputs=c2_files)
                        with gr.Column(scale=1):
                            c2_conf = gr.Slider(minimum=40, maximum=100, value=80, step=5, label="Seuil de confiance (%)")
                    c2_btn = gr.Button("🚀 Lancer Crop & Classify", variant="primary")
                    with gr.Row():
                        c2_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        c2_dl = gr.File(label="⬇️ Télécharger ZIP", visible=False)
                        c2_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_c2(files, conf):
                        for logs, zip_p, df in run_classify_mode(files, conf, "crop_and_classify"):
                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
                    c2_btn.click(fn_c2, inputs=[c2_files, c2_conf], outputs=[c2_log, c2_dl, c2_df])

                with gr.Tab("Labeliser"):
                    gr.Markdown("### Génération d'annotations YOLO (images/ et labels/)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            c3_files = gr.File(label="Upload Images ou PDFs", file_count="multiple", file_types=["image", ".pdf"], elem_classes=["horizontal-files"])
                            c3_files.change(fn=lambda f: gr.update(label=f"Upload Images ou PDFs ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images ou PDFs"), inputs=c3_files, outputs=c3_files)
                        with gr.Column(scale=1):
                            c3_conf = gr.Slider(minimum=40, maximum=100, value=80, step=5, label="Seuil de confiance (%)")
                    c3_btn = gr.Button("🚀 Générer Annotations YOLO", variant="primary")
                    with gr.Row():
                        c3_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        c3_dl = gr.File(label="⬇️ Télécharger ZIP (images + labels)", visible=False)
                        c3_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_c3(files, conf):
                        for logs, zip_p, df in run_classify_mode(files, conf, "yolo_annotation"):
                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
                    c3_btn.click(fn_c3, inputs=[c3_files, c3_conf], outputs=[c3_log, c3_dl, c3_df])

        # ── Tab 2 : Only Deskew ─────────────────────────────────────────────
        with gr.Tab("Redresser"):
            gr.Markdown("### Redressement automatique")
            with gr.Row():
                with gr.Column(scale=1):
                    dk_files = gr.File(
                        label="Upload Images pré-recadrées",
                        file_count="multiple",
                        file_types=["image"],
                        elem_classes=["horizontal-files"]
                    )
                    dk_files.change(fn=lambda f: gr.update(label=f"Upload Images pré-recadrées ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images pré-recadrées"), inputs=dk_files, outputs=dk_files)
                with gr.Column(scale=1):
                    dk_conf_deskew = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5,
                        label="Seuil de confiance redressement (%)",
                    )

            dk_run_btn = gr.Button("🚀 Lancer le redressement", variant="primary")

            with gr.Row():
                dk_log = gr.Textbox(
                    label="📋 Logs",
                    lines=12,
                    interactive=False,
                    placeholder="Les logs apparaîtront ici…",
                )
            with gr.Row():
                dk_download = gr.File(label="⬇️ Télécharger le ZIP des résultats", visible=False)
                dk_df = gr.Dataframe(label="📊 Résultats de redressement", visible=False)

            def deskew_and_show(files, conf_deskew):
                for logs, zip_path, df in run_deskew(files, conf_deskew):
                    yield (
                        logs,
                        gr.update(value=zip_path, visible=zip_path is not None),
                        df_update_with_count(df, "📊 Résultats de redressement"),
                    )

            dk_run_btn.click(
                fn=deskew_and_show,
                inputs=[dk_files, dk_conf_deskew],
                outputs=[dk_log, dk_download, dk_df],
            )

        # ── Tab 3 : Texte (OBB) ─────────────────────────────────────────────
        with gr.Tab("Détecteur d'identifiant"):
            with gr.Tabs():
                with gr.Tab("Détecter et découper"):
                    gr.Markdown("### Détecter, redresser et découper")
                    with gr.Row():
                        with gr.Column(scale=1):
                            obb_c1_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            obb_c1_files.change(fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"), inputs=obb_c1_files, outputs=obb_c1_files)
                        with gr.Column(scale=1):
                            obb_c1_conf = gr.Slider(minimum=10, maximum=100, value=25, step=5, label="Seuil de confiance (%)")
                            obb_c1_card_type = gr.Dropdown(choices=["CEDEAO", "CIP", "CNI", "PASSEPORT"], value=None, label="Type de carte")

                    obb_c1_btn = gr.Button("🚀 Lancer Crop & Resize", variant="primary")
                    with gr.Row():
                        obb_c1_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        obb_c1_dl = gr.File(label="⬇️ Télécharger ZIP")
                        obb_c1_df = gr.Dataframe(label="📊 Résultats")

                    def fn_obb_c1(files, conf, card_type):
                        for logs, zip_p, df in run_obb_mode(files, conf, "crop_resize", card_type=card_type):
                            yield logs, zip_p, df_update_with_count(df)
                    obb_c1_btn.click(fn_obb_c1, inputs=[obb_c1_files, obb_c1_conf, obb_c1_card_type], outputs=[obb_c1_log, obb_c1_dl, obb_c1_df])

                with gr.Tab("Générer Labels OBB"):
                    gr.Markdown("### Générer un dataset YOLO OBB (images letterboxées et labels)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            obb_c2_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            obb_c2_files.change(fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"), inputs=obb_c2_files, outputs=obb_c2_files)
                        with gr.Column(scale=1):
                            obb_c2_conf = gr.Slider(minimum=10, maximum=100, value=25, step=5, label="Seuil de confiance (%)")
                            with gr.Row():
                                obb_c2_w = gr.Number(label="Largeur cible (px)", value=640, precision=0)
                                obb_c2_h = gr.Number(label="Hauteur cible (px)", value=640, precision=0)

                    obb_c2_btn = gr.Button("Générer Dataset OBB", variant="primary")
                    with gr.Row():
                        obb_c2_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        obb_c2_dl = gr.File(label="⬇️ Télécharger ZIP (images + labels)")
                        obb_c2_df = gr.Dataframe(label="📊 Résultats")

                    def fn_obb_c2(files, conf, w, h):
                        for logs, zip_p, df in run_obb_mode(files, conf, "generate_labels", int(w), int(h), ""):
                            yield logs, zip_p, df_update_with_count(df)
                    obb_c2_btn.click(fn_obb_c2, inputs=[obb_c2_files, obb_c2_conf, obb_c2_w, obb_c2_h], outputs=[obb_c2_log, obb_c2_dl, obb_c2_df])

        # ── Tab 4 : Reconnaissance (OCR) ─────────────────────────────────────────────
        with gr.Tab("🔤 Reconnaissance"):

            with gr.Tabs():

                # ── Sous-onglet 1 : PARSeq Générale ───────────────────────────
                with gr.Tab("Parseq générale"):
                    gr.Markdown("### Extraire le texte avec PARSeq (poids généralistes)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ocr_gen_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            ocr_gen_files.change(
                                fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"),
                                inputs=ocr_gen_files, outputs=ocr_gen_files
                            )
                        with gr.Column(scale=1):
                            ocr_gen_conf = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Seuil de confiance minimum (%)")

                    ocr_gen_btn = gr.Button("🚀 Lancer OCR Générale (PARSeq)", variant="primary")
                    with gr.Row():
                        ocr_gen_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        ocr_gen_dl = gr.File(label="⬇️ Télécharger labels.csv", visible=False)
                        ocr_gen_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_ocr_gen(files, conf):
                        logs, csv_p, df = run_ocr_mode(files, conf, model_type="general")
                        return logs, gr.update(value=csv_p, visible=csv_p is not None), df_update_with_count(df)
                    ocr_gen_btn.click(fn_ocr_gen, inputs=[ocr_gen_files, ocr_gen_conf], outputs=[ocr_gen_log, ocr_gen_dl, ocr_gen_df])

                # ── Sous-onglet 2 : Reconnaissance Nom ──────────────────────────
                with gr.Tab("Reconnaissance nom"):
                    gr.Markdown("### Extraire les noms avec PARSeq (poids fine-tunés `Nicias/ocr_nom`)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ocr_nom_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            ocr_nom_files.change(
                                fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"),
                                inputs=ocr_nom_files, outputs=ocr_nom_files
                            )
                        with gr.Column(scale=1):
                            ocr_nom_conf = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Seuil de confiance minimum (%)")

                    ocr_nom_btn = gr.Button("🚀 Lancer OCR Nom (PARSeq)", variant="primary")
                    with gr.Row():
                        ocr_nom_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        ocr_nom_dl = gr.File(label="⬇️ Télécharger labels.csv", visible=False)
                        ocr_nom_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_ocr_nom(files, conf):
                        logs, csv_p, df = run_ocr_mode(files, conf, model_type="nom")
                        return logs, gr.update(value=csv_p, visible=csv_p is not None), df_update_with_count(df)
                    ocr_nom_btn.click(fn_ocr_nom, inputs=[ocr_nom_files, ocr_nom_conf], outputs=[ocr_nom_log, ocr_nom_dl, ocr_nom_df])


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    # On HF Spaces, just call launch() with no args — HF handles routing.
    # Locally, bind to 0.0.0.0:7860 for Docker / LAN access.
    if os.getenv("SPACE_ID"):
        demo.launch()
    else:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
