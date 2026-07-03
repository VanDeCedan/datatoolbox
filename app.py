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
from recognizer import ParseqNomRecognizer, ParseqDateRecognizer
from pdf_utils import pdf_to_images, count_pdf_pages

# ---------------------------------------------------------------------------
# Model loaders (lazy, cached at module level)
# ---------------------------------------------------------------------------
_detector: Optional[CardDetector] = None
_deskewer: Optional[CardDeskewer] = None
_obb_detector: Optional[ObbDetector] = None
_parseq_nom: Optional[ParseqNomRecognizer] = None
_parseq_date: Optional[ParseqDateRecognizer] = None


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


def get_parseq_nom_recognizer() -> ParseqNomRecognizer:
    global _parseq_nom
    if _parseq_nom is None:
        _parseq_nom = ParseqNomRecognizer()
    return _parseq_nom


def get_parseq_date_recognizer() -> ParseqDateRecognizer:
    global _parseq_date
    if _parseq_date is None:
        _parseq_date = ParseqDateRecognizer()
    return _parseq_date





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
        return "⚠️ Aucun fichier uploadé.", None, None

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

        for uf in uploaded_files:
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
        return f"❌ Erreur inattendue : {exc}", None, None


# ---------------------------------------------------------------------------
# Tab 2 — Only Deskew
# ---------------------------------------------------------------------------
def run_deskew(
    uploaded_files,
    conf_deskew_pct: int,
) -> tuple[str, Optional[str], Optional[pd.DataFrame]]:
    if not uploaded_files:
        return "⚠️ Aucun fichier uploadé.", None, None

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

        for uf in uploaded_files:
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

        if not all_records:
            return "\n".join(logs), None, None

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_deskew.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")
        log(f"🎉 Terminé ! {len(all_records)} image(s) redressée(s).")

        df = pd.DataFrame(all_records)
        return "\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        return f"❌ Erreur inattendue : {exc}", None, None


# ---------------------------------------------------------------------------
# Tab 4 — Reconnaissance (OCR) Logic
# ---------------------------------------------------------------------------
def run_ocr_mode(
    uploaded_files,
    conf_threshold_pct: int,
    model_type: str = "nom",
):
    if not uploaded_files:
        yield "⚠️ Aucun fichier uploadé.", None, None
        return

    logs = []
    def log(msg: str): logs.append(msg)

    label_map = {
        "nom": "PARSeq Nom",
        "date": "PARSeq Date Expiration"
    }
    label = label_map.get(model_type, "PARSeq Nom")
    log(f"🔮 Chargement du modèle OCR ({label})…")
    yield "\n".join(logs), None, None

    try:
        if model_type == "nom":
            recognizer = get_parseq_nom_recognizer()
        else:
            recognizer = get_parseq_date_recognizer()
    except Exception as e:
        yield f"❌ Erreur de chargement du modèle: {e}", None, None
        return

    tmp_root = tempfile.mkdtemp()
    all_records = []
    csv_data = []

    try:
        tmp_path = Path(tmp_root)
        total = len(uploaded_files)
        UPDATE_INTERVAL = 50

        for idx, uf in enumerate(uploaded_files):
            fpath = _get_path(uf)
            fname = Path(fpath).name

            if idx % UPDATE_INTERVAL == 0:
                log(f"🔄 Progrès : {idx}/{total} images traitées...")
                yield "\n".join(logs), None, pd.DataFrame(all_records) if all_records else None

            try:
                pil_img = Image.open(fpath).convert("RGB")

                # Predict
                text, conf = recognizer.recognize(pil_img)
                text_clean = text.strip() if text else ""

                if not text_clean or conf < (conf_threshold_pct / 100.0):
                    if total <= 10:
                        log(f"  ⚠️ Ignoré: '{text_clean}' (conf: {conf:.2f})")
                    continue

                if total <= 10:
                    log(f"  ✅ Texte: '{text_clean}' (Confiance: {conf:.2f})")

                all_records.append({"Fichier": fname, "Texte": text_clean, "Confiance": f"{conf:.2f}"})
                csv_data.append(f"{fname},{text_clean}")

            except Exception as exc:
                log(f"  ❌ Erreur sur {fname}: {exc}")

            if (idx + 1) % 500 == 0:
                import gc
                gc.collect()
                # Memory clearing not needed for ONNX
                pass

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


def run_merge_labels(image_paths: list[str], csv_paths: list[str], include_images: bool = False) -> tuple[str, str | None, pd.DataFrame | None]:
    if not image_paths:
        return "⚠️ Aucun fichier image dans le lot.", None, None
    if not csv_paths:
        return "⚠️ Aucun fichier CSV chargé.", None, None

    logs = []
    def log(msg: str):
        logs.append(msg)
        print(msg)

    log(f"📋 Début de la fusion. Lot d'images : {len(image_paths)} fichiers, CSV de labels : {len(csv_paths)} fichiers.")
    
    # 1. Lire et fusionner tous les CSV
    merged_labels = {}
    
    for csv_file in csv_paths:
        fname_csv = Path(csv_file).name
        try:
            # Essayer de lire le CSV
            # On essaie d'abord la détection automatique du délimiteur via pandas
            try:
                df = pd.read_csv(csv_file, sep=None, engine='python', encoding='utf-8', dtype=str)
            except Exception:
                df = pd.read_csv(csv_file, sep=',', encoding='utf-8', dtype=str)
            
            # Identifier les colonnes
            cols = [str(c).strip().lower() for c in df.columns]
            col_img = None
            col_txt = None
            
            # Recherche heuristique des colonnes image et texte
            img_candidates = ["image", "fichier", "filename", "file", "img", "id"]
            txt_candidates = ["text", "texte", "label", "value", "valeur", "class", "classe"]
            
            for c_name in df.columns:
                c_clean = str(c_name).strip().lower()
                if any(cand in c_clean for cand in img_candidates):
                    col_img = c_name
                    break
            if not col_img:
                # Si aucun candidat, on prend la 1ère colonne qui a des chaînes se terminant par une extension d'image
                for c_name in df.columns:
                    sample = df[c_name].dropna().astype(str).head(5)
                    if any(s.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) for s in sample):
                        col_img = c_name
                        break
            
            for c_name in df.columns:
                c_clean = str(c_name).strip().lower()
                if any(cand in c_clean for cand in txt_candidates) and c_name != col_img:
                    col_txt = c_name
                    break
            
            # Fallback par index
            if not col_img and len(df.columns) >= 1:
                col_img = df.columns[0]
            if not col_txt and len(df.columns) >= 2:
                col_txt = df.columns[1]
                
            if not col_img or not col_txt:
                log(f"  ⚠️ Impossible d'identifier les colonnes d'images ou de textes dans {fname_csv}. Colonnes trouvées : {list(df.columns)}")
                continue
                
            log(f"  📖 Lecture de {fname_csv} (Image: '{col_img}', Label: '{col_txt}')")
            
            # Remplir le dictionnaire
            rows_processed = 0
            for _, row in df.iterrows():
                val_img = str(row[col_img]).strip()
                val_txt = str(row[col_txt]).strip() if pd.notna(row[col_txt]) else ""
                
                if val_img:
                    # On garde le nom de fichier propre
                    img_name = Path(val_img).name
                    # Si on n'a pas encore de label ou s'il est vide, on prend celui-ci
                    if img_name not in merged_labels or not merged_labels[img_name]:
                        merged_labels[img_name] = val_txt
                    rows_processed += 1
            log(f"    ✅ {rows_processed} lignes traitées de {fname_csv}")
            
        except Exception as e:
            log(f"  ❌ Erreur lors de la lecture du CSV {fname_csv} : {e}")

    log(f"📊 Total des labels chargés depuis les CSV : {len(merged_labels)} images uniques référencées.")

    # 2. Associer les images physiques
    matched_images = []
    missing_images = []
    
    # Mapper pour retrouver le chemin physique par rapport au nom de fichier
    img_map = {}
    for p in image_paths:
        name = Path(p).name
        if name not in img_map:
            img_map[name] = p

    for img_name, full_path in img_map.items():
        if img_name in merged_labels:
            matched_images.append((img_name, full_path, merged_labels[img_name]))
        else:
            missing_images.append(img_name)

    log(f"📈 Images retrouvées dans les labels : {len(matched_images)} / {len(img_map)}")
    log(f"📉 Images orphelines (absentes des labels) : {len(missing_images)}")

    # 3. Préparer l'output zip et les fichiers temporaires
    tmp_root = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "fusion"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier les images retrouvées uniquement si demandé
        if include_images:
            log(f"📦 Copie de {len(matched_images)} images physiques dans le lot de sortie…")
            for img_name, src_path, label in matched_images:
                dest_img_path = out_dir / img_name
                try:
                    shutil.copy2(src_path, dest_img_path)
                except Exception as copy_err:
                    # Ne pas faire planter toute la fusion pour une erreur de copie
                    pass
        else:
            log("📝 Mode CSV uniquement (les images physiques ne seront pas incluses dans le ZIP pour économiser la mémoire et le temps de traitement).")
            
        # Créer le CSV unique
        out_csv_path = out_dir / "labels.csv"
        rows_to_save = [{"image": item[0], "text": item[2]} for item in matched_images]
        df_out = pd.DataFrame(rows_to_save)
        df_out.to_csv(out_csv_path, index=False, encoding="utf-8")
        
        # Zipper le dossier fusion
        zip_path = tmp_path / "fusion_labels.zip"
        build_zip_to_file(out_dir, zip_path)
        
        # Créer le dataframe de retour (pour affichage Gradio)
        df_display = pd.DataFrame([{"Fichier": item[0], "Label": item[2]} for item in matched_images])
        
        # Ajouter le rapport détaillé pour les images manquantes
        if missing_images:
            log("\n⚠️ RAPPORT : Images qui n'ont pas été retrouvées dans les fichiers CSV :")
            # Limiter l'affichage dans les logs si la liste est gigantesque (ex: 20 000 images) pour éviter de saturer la UI
            max_log_missing = 100
            for m_img in sorted(missing_images)[:max_log_missing]:
                log(f"  - {m_img}")
            if len(missing_images) > max_log_missing:
                log(f"  ... et {len(missing_images) - max_log_missing} autres images.")
        else:
            log("\n🎉 Succès : Toutes les images ont été retrouvées dans au moins un CSV !")

        return "\n".join(logs), str(zip_path), df_display

    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return f"❌ Erreur lors de la création du ZIP de sortie : {e}\n\nLogs de fusion :\n" + "\n".join(logs), None, None


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
        return "⚠️ Aucun fichier uploadé.", None, None
        
    if mode == "crop_resize" and not card_type:
        return "⚠️ Erreur: Le type de carte est obligatoire pour lancer le découpage.", None, None

    conf = conf_pct / 100.0
    logs = []
    def log(msg: str): logs.append(msg)

    log("🔮 Chargement du modèle OBB…")
    try:
        detector = get_obb_detector(conf)
    except Exception as e:
        return f"❌ Erreur de chargement du modèle: {e}", None, None

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

        for uf in uploaded_files:
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

        if not all_records:
            log("⚠️ Aucun résultat.")
            return "\n".join(logs), None, None

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_obb.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")

        df = pd.DataFrame(all_records)
        return "\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        return f"❌ Erreur inattendue : {exc}", None, None


def apply_camera_grain(pil_img: Image.Image) -> Image.Image:
    import numpy as np
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    noise = np.random.normal(0, 8, (h, w, c))
    noisy_img = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(noisy_img)
    del img_np, noise, noisy_img
    return out_img


def apply_salt_pepper_noise(pil_img: Image.Image, amount: float = 0.01) -> Image.Image:
    import numpy as np
    img_np = np.array(pil_img)
    h, w, c = img_np.shape
    noisy_img = img_np.copy()
    num_pixels = int(amount * h * w)
    
    salt_y = np.random.randint(0, h, num_pixels)
    salt_x = np.random.randint(0, w, num_pixels)
    noisy_img[salt_y, salt_x] = 255
    
    pepper_y = np.random.randint(0, h, num_pixels)
    pepper_x = np.random.randint(0, w, num_pixels)
    noisy_img[pepper_y, pepper_x] = 0
    
    out_img = Image.fromarray(noisy_img)
    del img_np, noisy_img, salt_y, salt_x, pepper_y, pepper_x
    return out_img


def apply_blur(pil_img: Image.Image, ksize: int = 5) -> Image.Image:
    import cv2
    import numpy as np
    img_np = np.array(pil_img)
    blurred = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
    out_img = Image.fromarray(blurred)
    del img_np, blurred
    return out_img


def flip_labels_horizontal(coords: list[float]) -> list[float]:
    # x' = 1.0 - x, y' = y
    # P'_1 = (1 - x_2, y_2)
    # P'_2 = (1 - x_1, y_1)
    # P'_3 = (1 - x_4, y_4)
    # P'_4 = (1 - x_3, y_3)
    return [
        1.0 - coords[2], coords[3],
        1.0 - coords[0], coords[1],
        1.0 - coords[6], coords[7],
        1.0 - coords[4], coords[5]
    ]


def flip_labels_vertical(coords: list[float]) -> list[float]:
    # x' = x, y' = 1.0 - y
    # P'_1 = (x_4, 1 - y_4)
    # P'_2 = (x_3, 1 - y_3)
    # P'_3 = (x_2, 1 - y_2)
    # P'_4 = (x_1, 1 - y_1)
    return [
        coords[6], 1.0 - coords[7],
        coords[4], 1.0 - coords[5],
        coords[2], 1.0 - coords[3],
        coords[0], 1.0 - coords[1]
    ]


def flip_labels_both(coords: list[float]) -> list[float]:
    # x' = 1.0 - x, y' = 1.0 - y
    # P'_1 = (1 - x_3, 1 - y_3)
    # P'_2 = (1 - x_4, 1 - y_4)
    # P'_3 = (1 - x_1, 1 - y_1)
    # P'_4 = (1 - x_2, 1 - y_2)
    return [
        1.0 - coords[4], 1.0 - coords[5],
        1.0 - coords[6], 1.0 - coords[7],
        1.0 - coords[0], 1.0 - coords[1],
        1.0 - coords[2], 1.0 - coords[3]
    ]


def run_get_labels_images(
    uploaded_images,
    uploaded_labels,
    aug_flip_h: bool,
    aug_flip_v: bool,
    aug_flip_hv: bool,
    aug_grain: bool,
    aug_noise: bool,
    aug_blur: bool,
    num_augs: int,
    split_enabled: bool,
    train_pct: float,
    val_pct: float,
    test_pct: float,
):
    import random
    import gc
    
    if not uploaded_images:
        yield "⚠️ Aucun fichier image importé.", None, None
        return
    if not uploaded_labels:
        yield "⚠️ Aucun fichier label (.txt) importé.", None, None
        return

    logs = []
    def log(msg: str):
        logs.append(msg)

    log("🔍 Analyse et appariement des fichiers…")
    yield "\n".join(logs), None, None
    
    label_dict = {}
    for lf in uploaded_labels:
        lpath = _get_path(lf)
        lstem = Path(lpath).stem
        label_dict[lstem] = lpath
        
    matched_pairs = []
    for imgf in uploaded_images:
        ipath = _get_path(imgf)
        istem = Path(ipath).stem
        if istem in label_dict:
            matched_pairs.append((ipath, label_dict[istem], istem))
        else:
            log(f"⚠️ Image ignorée (aucun label correspondant) : {Path(ipath).name}")
            
    if not matched_pairs:
        yield "❌ Aucun appariement trouvé entre les images et les fichiers labels (.txt). Assurez-vous qu'ils partagent le même nom de fichier (ex: image1.jpg et image1.txt).", None, None
        return

    log(f"✅ Trouvé {len(matched_pairs)} paires d'images et labels correspondants.")
    yield "\n".join(logs), None, None

    if split_enabled:
        total = train_pct + val_pct + test_pct
        if total <= 0:
            train_pct, val_pct, test_pct = 70.0, 15.0, 15.0
            total = 100.0
        if abs(total - 100.0) > 0.01:
            log(f"⚠️ Les pourcentages ({train_pct}%, {val_pct}%, {test_pct}%) ne somment pas à 100%. Normalisation automatique…")
            train_pct = (train_pct / total) * 100.0
            val_pct = (val_pct / total) * 100.0
            test_pct = (test_pct / total) * 100.0
            
        log(f"📊 Répartition demandée : Train={train_pct:.1f}%, Valid={val_pct:.1f}%, Test={test_pct:.1f}%")
        
        shuffled_pairs = matched_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        n = len(shuffled_pairs)
        train_count = int(round(n * (train_pct / 100.0)))
        val_count = int(round(n * (val_pct / 100.0)))
        test_count = max(0, n - train_count - val_count)
        
        train_pairs = shuffled_pairs[:train_count]
        val_pairs = shuffled_pairs[train_count:train_count+val_count]
        test_pairs = shuffled_pairs[train_count+val_count:]
        
        splits = {
            "train": train_pairs,
            "valid": val_pairs,
            "test": test_pairs
        }
        
        log(f"   - Train : {len(train_pairs)} paires d'origine")
        log(f"   - Valid : {len(val_pairs)} paires d'origine")
        log(f"   - Test  : {len(test_pairs)} paires d'origine")
    else:
        splits = {
            "": matched_pairs
        }
        log(f"📊 Mode sans split : Toutes les {len(matched_pairs)} paires seront dans le dossier racine.")

    yield "\n".join(logs), None, None

    tmp_root = tempfile.mkdtemp()
    all_records = []
    
    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "output_dataset"
        out_dir.mkdir()
        
        for split_name, pairs in splits.items():
            if not pairs:
                continue
            if split_name:
                split_dir = out_dir / split_name
                split_dir.mkdir(exist_ok=True)
                (split_dir / "images").mkdir(exist_ok=True)
                (split_dir / "labels").mkdir(exist_ok=True)
            else:
                (out_dir / "images").mkdir(exist_ok=True)
                (out_dir / "labels").mkdir(exist_ok=True)
            
        log("🚀 Génération des images et application des augmentations…")
        yield "\n".join(logs), None, None
        
        total_pairs = len(matched_pairs)
        yield_interval = 1 if total_pairs <= 50 else (10 if total_pairs <= 500 else 100)
        processed_count = 0
        
        for split_name, pairs in splits.items():
            if not pairs:
                continue
            if split_name:
                split_img_dir = out_dir / split_name / "images"
                split_lbl_dir = out_dir / split_name / "labels"
            else:
                split_img_dir = out_dir / "images"
                split_lbl_dir = out_dir / "labels"
            
            for img_path, lbl_path, stem in pairs:
                processed_count += 1
                log(f"  - [{processed_count}/{total_pairs}] Traitement de {stem}...")
                
                try:
                    with Image.open(img_path).convert("RGB") as pil_img:
                        obb_labels = []
                        try:
                            with open(lbl_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 9:
                                        class_id = parts[0]
                                        coords = [float(x) for x in parts[1:9]]
                                        obb_labels.append((class_id, coords))
                        except Exception as e:
                            log(f"  ❌ Impossible de charger le label {Path(lbl_path).name} : {e}")
                            continue
                        
                        orig_img_name = f"{stem}.jpg"
                        orig_lbl_name = f"{stem}.txt"
                        pil_img.save(str(split_img_dir / orig_img_name), "JPEG", quality=95)
                        with open(split_lbl_dir / orig_lbl_name, "w", encoding="utf-8") as f:
                            for cid, coords in obb_labels:
                                coords_str = " ".join(f"{c:.6f}" for c in coords)
                                f.write(f"{cid} {coords_str}\n")
                                
                        all_records.append({
                            "Nom de base": stem,
                            "Type": "Original",
                            "Split": split_name if split_name else "root",
                            "Statut": "OK"
                        })
                        
                        enabled_transforms = []
                        if aug_flip_h:
                            enabled_transforms.append("flip_h")
                        if aug_flip_v:
                            enabled_transforms.append("flip_v")
                        if aug_flip_hv:
                            enabled_transforms.append("flip_hv")
                        if aug_grain:
                            enabled_transforms.append("grain")
                        if aug_noise:
                            enabled_transforms.append("noise")
                        if aug_blur:
                            enabled_transforms.append("blur")
                        
                        for k in range(num_augs):
                            aug_img = pil_img.copy()
                            aug_labels = obb_labels.copy()
                            applied_augs_names = []
                            
                            if enabled_transforms:
                                num_to_apply = random.randint(1, min(3, len(enabled_transforms)))
                                chosen_transforms = random.sample(enabled_transforms, num_to_apply)
                                
                                flip_type = None
                                flips_in_choice = [t for t in chosen_transforms if t in ["flip_h", "flip_v", "flip_hv"]]
                                if flips_in_choice:
                                    flip_type = random.choice(flips_in_choice)
                                    
                                if flip_type == "flip_h":
                                    tmp_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                                    aug_img.close()
                                    aug_img = tmp_img
                                    new_aug_labels = []
                                    for cid, coords in aug_labels:
                                        new_coords = flip_labels_horizontal(coords)
                                        new_aug_labels.append((cid, new_coords))
                                    aug_labels = new_aug_labels
                                    applied_augs_names.append("FlipH")
                                elif flip_type == "flip_v":
                                    tmp_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
                                    aug_img.close()
                                    aug_img = tmp_img
                                    new_aug_labels = []
                                    for cid, coords in aug_labels:
                                        new_coords = flip_labels_vertical(coords)
                                        new_aug_labels.append((cid, new_coords))
                                    aug_labels = new_aug_labels
                                    applied_augs_names.append("FlipV")
                                elif flip_type == "flip_hv":
                                    tmp_img = aug_img.transpose(Image.ROTATE_180)
                                    aug_img.close()
                                    aug_img = tmp_img
                                    new_aug_labels = []
                                    for cid, coords in aug_labels:
                                        new_coords = flip_labels_both(coords)
                                        new_aug_labels.append((cid, new_coords))
                                    aug_labels = new_aug_labels
                                    applied_augs_names.append("FlipHV")
                                    
                                if "grain" in chosen_transforms:
                                    tmp_img = apply_camera_grain(aug_img)
                                    aug_img.close()
                                    aug_img = tmp_img
                                    applied_augs_names.append("Grain")
                                if "noise" in chosen_transforms:
                                    if random.random() < 0.5:
                                        tmp_img = apply_salt_pepper_noise(aug_img, amount=random.uniform(0.008, 0.02))
                                        aug_img.close()
                                        aug_img = tmp_img
                                        applied_augs_names.append("SaltPepper")
                                    else:
                                        import numpy as np
                                        img_np = np.array(aug_img)
                                        noise = np.random.normal(0, random.uniform(10, 20), img_np.shape)
                                        noisy_img = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                                        tmp_img = Image.fromarray(noisy_img)
                                        aug_img.close()
                                        aug_img = tmp_img
                                        applied_augs_names.append("GaussianNoise")
                                        
                                if "blur" in chosen_transforms:
                                    ksize = random.choice([3, 5])
                                    tmp_img = apply_blur(aug_img, ksize=ksize)
                                    aug_img.close()
                                    aug_img = tmp_img
                                    applied_augs_names.append(f"BlurK{ksize}")
                            else:
                                applied_augs_names.append("None")
                                
                            suffix = f"_aug_{k+1}"
                            aug_img_name = f"{stem}{suffix}.jpg"
                            aug_lbl_name = f"{stem}{suffix}.txt"
                            
                            aug_img.save(str(split_img_dir / aug_img_name), "JPEG", quality=95)
                            aug_img.close()
                            
                            with open(split_lbl_dir / aug_lbl_name, "w", encoding="utf-8") as f:
                                for cid, coords in aug_labels:
                                    coords_str = " ".join(f"{c:.6f}" for c in coords)
                                    f.write(f"{cid} {coords_str}\n")
                                    
                            all_records.append({
                                "Nom de base": stem,
                                "Type": f"Augmenté ({', '.join(applied_augs_names)})",
                                "Split": split_name if split_name else "root",
                                "Statut": "OK"
                            })
                            
                except Exception as e:
                    log(f"  ❌ Impossible de charger l'image {Path(img_path).name} : {e}")
                    continue

                # Force garbage collection after every image to keep memory flat
                gc.collect()

                if processed_count % yield_interval == 0 or processed_count == total_pairs:
                    yield "\n".join(logs), None, None
                    
        if not all_records:
            log("⚠️ Aucun fichier généré.")
            yield "\n".join(logs), None, None
            return
            
        log("📦 Génération du ZIP…")
        yield "\n".join(logs), None, None
        
        zip_path = tmp_path / "dataset_yolo_obb_augmented.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / (1024 * 1024):.2f} MB")
        
        df = pd.DataFrame(all_records)
        log(f"📊 {len(df)} lignes de résultats générées (affichage des 100 premières lignes dans le tableau).")
        yield "\n".join(logs), str(zip_path), df.head(100)

    except Exception as exc:
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        yield f"❌ Erreur inattendue : {exc}", None, None


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
                        logs, zip_p, df = run_classify_mode(files, conf, "classification_only")
                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
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
                        logs, zip_p, df = run_classify_mode(files, conf, "crop_and_classify")
                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
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
                        logs, zip_p, df = run_classify_mode(files, conf, "yolo_annotation")
                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)
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
                logs, zip_path, df = run_deskew(files, conf_deskew)
                return (
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
                        logs, zip_p, df = run_obb_mode(files, conf, "crop_resize", card_type=card_type)
                        return logs, zip_p, df_update_with_count(df)
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
                        logs, zip_p, df = run_obb_mode(files, conf, "generate_labels", int(w), int(h), "")
                        return logs, zip_p, df_update_with_count(df)
                    obb_c2_btn.click(fn_obb_c2, inputs=[obb_c2_files, obb_c2_conf, obb_c2_w, obb_c2_h], outputs=[obb_c2_log, obb_c2_dl, obb_c2_df])

                with gr.Tab("Get labels images"):
                    gr.Markdown("### Appliquer des augmentations et split sur des datasets d'images et labels OBB")
                    
                    obb_c3_images_state = gr.State([])
                    obb_c3_labels_state = gr.State([])

                    with gr.Row():
                        with gr.Column(scale=1):
                            obb_c3_images = gr.File(label="Upload Images (cumulatif)", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            with gr.Row():
                                obb_c3_img_dir = gr.Textbox(label="Ou dossier d'images local", placeholder="Ex: C:\\images")
                                obb_c3_img_scan_btn = gr.Button("🔍 Scanner images", variant="secondary")
                            with gr.Row():
                                obb_c3_images_status = gr.Markdown("**Images accumulées :** 0")
                                obb_c3_images_clear = gr.Button("🗑️ Vider les images", variant="secondary")
                        with gr.Column(scale=1):
                            obb_c3_labels = gr.File(label="Upload Labels (OBB .txt) (cumulatif)", file_count="multiple", file_types=[".txt"], elem_classes=["horizontal-files"])
                            with gr.Row():
                                obb_c3_lbl_dir = gr.Textbox(label="Ou dossier de labels local", placeholder="Ex: C:\\labels")
                                obb_c3_lbl_scan_btn = gr.Button("🔍 Scanner labels", variant="secondary")
                            with gr.Row():
                                obb_c3_labels_status = gr.Markdown("**Labels accumulés :** 0")
                                obb_c3_labels_clear = gr.Button("🗑️ Vider les labels", variant="secondary")

                    def add_c3_images(new_files, current_state):
                        if not new_files:
                            yield current_state, f"**Images accumulées :** {len(current_state)}"
                            return
                        import gc
                        existing_names = {Path(p).name for p in current_state}
                        batch_size = 50
                        total_new = len(new_files)
                        for i in range(0, total_new, batch_size):
                            batch = new_files[i:i+batch_size]
                            for f in batch:
                                path_str = _get_path(f)
                                name = Path(path_str).name
                                if name not in existing_names:
                                    current_state.append(path_str)
                                    existing_names.add(name)
                            gc.collect()
                            yield current_state, f"**Images accumulées :** {len(current_state)} (Chargement : {min(i+batch_size, total_new)}/{total_new})"
                        yield current_state, f"**Images accumulées :** {len(current_state)}"

                    def add_c3_labels(new_files, current_state):
                        if not new_files:
                            yield current_state, f"**Labels accumulés :** {len(current_state)}"
                            return
                        import gc
                        existing_names = {Path(p).name for p in current_state}
                        batch_size = 50
                        total_new = len(new_files)
                        for i in range(0, total_new, batch_size):
                            batch = new_files[i:i+batch_size]
                            for f in batch:
                                path_str = _get_path(f)
                                name = Path(path_str).name
                                if name not in existing_names:
                                    current_state.append(path_str)
                                    existing_names.add(name)
                            gc.collect()
                            yield current_state, f"**Labels accumulés :** {len(current_state)} (Chargement : {min(i+batch_size, total_new)}/{total_new})"
                        yield current_state, f"**Labels accumulés :** {len(current_state)}"

                    def scan_c3_img_dir(dir_path, current_state):
                        if not dir_path or not os.path.exists(dir_path):
                            yield current_state, f"⚠️ Le dossier '{dir_path}' n'existe pas."
                            return
                        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                        found_files = []
                        try:
                            with os.scandir(dir_path) as it:
                                for entry in it:
                                    if entry.is_file():
                                        ext = Path(entry.name).suffix.lower()
                                        if ext in valid_exts:
                                            found_files.append(entry.path)
                        except Exception as e:
                            yield current_state, f"❌ Erreur lors du scan : {e}"
                            return
                        if not found_files:
                            yield current_state, f"⚠️ Aucune image trouvée dans '{dir_path}'."
                            return
                        
                        yield from add_c3_images(found_files, current_state)

                    def scan_c3_lbl_dir(dir_path, current_state):
                        if not dir_path or not os.path.exists(dir_path):
                            yield current_state, f"⚠️ Le dossier '{dir_path}' n'existe pas."
                            return
                        found_files = []
                        try:
                            with os.scandir(dir_path) as it:
                                for entry in it:
                                    if entry.is_file():
                                        ext = Path(entry.name).suffix.lower()
                                        if ext == '.txt':
                                            found_files.append(entry.path)
                        except Exception as e:
                            yield current_state, f"❌ Erreur lors du scan : {e}"
                            return
                        if not found_files:
                            yield current_state, f"⚠️ Aucun fichier .txt trouvé dans '{dir_path}'."
                            return
                        
                        yield from add_c3_labels(found_files, current_state)

                    obb_c3_images.change(
                        fn=add_c3_images,
                        inputs=[obb_c3_images, obb_c3_images_state],
                        outputs=[obb_c3_images_state, obb_c3_images_status]
                    )
                    obb_c3_labels.change(
                        fn=add_c3_labels,
                        inputs=[obb_c3_labels, obb_c3_labels_state],
                        outputs=[obb_c3_labels_state, obb_c3_labels_status]
                    )
                    
                    obb_c3_img_scan_btn.click(
                        fn=scan_c3_img_dir,
                        inputs=[obb_c3_img_dir, obb_c3_images_state],
                        outputs=[obb_c3_images_state, obb_c3_images_status]
                    )
                    obb_c3_lbl_scan_btn.click(
                        fn=scan_c3_lbl_dir,
                        inputs=[obb_c3_lbl_dir, obb_c3_labels_state],
                        outputs=[obb_c3_labels_state, obb_c3_labels_status]
                    )

                    obb_c3_images_clear.click(
                        fn=lambda: ([], "**Images accumulées :** 0"),
                        inputs=[],
                        outputs=[obb_c3_images_state, obb_c3_images_status]
                    )
                    obb_c3_labels_clear.click(
                        fn=lambda: ([], "**Labels accumulés :** 0"),
                        inputs=[],
                        outputs=[obb_c3_labels_state, obb_c3_labels_status]
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Options d'Augmentations géométriques (Flipping)")
                            aug_flip_h = gr.Checkbox(label="Flip Horizontal", value=True)
                            aug_flip_v = gr.Checkbox(label="Flip Vertical", value=False)
                            aug_flip_hv = gr.Checkbox(label="Both (H+V) Flip", value=False)
                        with gr.Column(scale=1):
                            gr.Markdown("#### Options d'Augmentations de rendu")
                            aug_grain = gr.Checkbox(label="Camera Grain", value=True)
                            aug_noise = gr.Checkbox(label="Gaussian & Salt/Pepper Noise", value=True)
                            aug_blur = gr.Checkbox(label="Gaussian Blur", value=True)
                    
                    with gr.Row():
                        num_augs = gr.Slider(minimum=1, maximum=20, value=3, step=1, label="Nombre d'augmentations à générer par image")

                    with gr.Group():
                        split_enabled = gr.Checkbox(label="Activer le Split Dataset (Train / Valid / Test)", value=True)
                        with gr.Row() as split_row:
                            train_pct = gr.Slider(minimum=0, maximum=100, value=70, step=1, label="Train (%)")
                            val_pct = gr.Slider(minimum=0, maximum=100, value=15, step=1, label="Valid (%)")
                            test_pct = gr.Slider(minimum=0, maximum=100, value=15, step=1, label="Test (%)")

                    def toggle_split_visibility(is_checked):
                        return gr.update(visible=is_checked)
                    split_enabled.change(fn=toggle_split_visibility, inputs=split_enabled, outputs=split_row)

                    obb_c3_btn = gr.Button("⚡ Générer Dataset augmenté & packagé", variant="primary")
                    
                    with gr.Row():
                        obb_c3_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        obb_c3_dl = gr.File(label="⬇️ Télécharger ZIP (Dataset YOLO OBB)")
                        obb_c3_df = gr.Dataframe(label="📊 Résultats")

                    def fn_obb_c3(images, labels, flip_h, flip_v, flip_hv, grain, noise, blur, count, split_on, train, val, test):
                        for logs, zip_p, df in run_get_labels_images(
                            images, labels, flip_h, flip_v, flip_hv, grain, noise, blur, int(count), split_on, train, val, test
                        ):
                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df, "📊 Résultats Dataset augmenté")
                        
                    obb_c3_btn.click(
                        fn=fn_obb_c3,
                        inputs=[
                            obb_c3_images_state, obb_c3_labels_state,
                            aug_flip_h, aug_flip_v, aug_flip_hv,
                            aug_grain, aug_noise, aug_blur,
                            num_augs,
                            split_enabled, train_pct, val_pct, test_pct
                        ],
                        outputs=[obb_c3_log, obb_c3_dl, obb_c3_df]
                    )

        # ── Tab 4 : Reconnaissance (OCR) ─────────────────────────────────────────────
        with gr.Tab("🔤 Reconnaissance"):
            with gr.Tabs():
                # ── Sous-onglet 1 : Reconnaissance Nom ──────────────────────────
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
                        for logs, csv_p, df in run_ocr_mode(files, conf, model_type="nom"):
                            yield logs, gr.update(value=csv_p, visible=csv_p is not None), df_update_with_count(df)
                    ocr_nom_btn.click(fn_ocr_nom, inputs=[ocr_nom_files, ocr_nom_conf], outputs=[ocr_nom_log, ocr_nom_dl, ocr_nom_df])


                # ── Sous-onglet 3 : Reconnaissance Date Expiration ──────────────
                with gr.Tab("Reconnaissance date expiration"):
                    gr.Markdown("### Extraire les dates d'expiration avec PARSeq (poids fine-tunés `Nicias/ocr_carte_date_expiration`)")
                    with gr.Row():
                        with gr.Column(scale=1):
                            ocr_date_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                            ocr_date_files.change(
                                fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"),
                                inputs=ocr_date_files, outputs=ocr_date_files
                            )
                        with gr.Column(scale=1):
                            ocr_date_conf = gr.Slider(minimum=0, maximum=100, value=50, step=5, label="Seuil de confiance minimum (%)")

                    ocr_date_btn = gr.Button("🚀 Lancer OCR Date Expiration (PARSeq)", variant="primary")
                    with gr.Row():
                        ocr_date_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                    with gr.Row():
                        ocr_date_dl = gr.File(label="⬇️ Télécharger labels.csv", visible=False)
                        ocr_date_df = gr.Dataframe(label="📊 Résultats", visible=False)

                    def fn_ocr_date(files, conf):
                        for logs, csv_p, df in run_ocr_mode(files, conf, model_type="date"):
                            yield logs, gr.update(value=csv_p, visible=csv_p is not None), df_update_with_count(df)
                    ocr_date_btn.click(fn_ocr_date, inputs=[ocr_date_files, ocr_date_conf], outputs=[ocr_date_log, ocr_date_dl, ocr_date_df])

        # ── Tab 5 : Lots de fichiers ──────────────────────────────────────────
        with gr.Tab("📦 Lots de fichiers"):
            with gr.Tabs():
                with gr.Tab("Fusionner labels CSV"):
                    gr.Markdown("### Fusionner des labels CSV avec des lots d'images")
                    
                    # Gradio States to accumulate file paths
                    batch_img_state = gr.State([])
                    batch_csv_state = gr.State([])

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📷 Lot d'images")
                            batch_img_upload = gr.File(
                                label="Ajouter des images au lot (cumulatif)",
                                file_count="multiple",
                                file_types=["image"],
                                elem_classes=["horizontal-files"]
                            )
                            with gr.Row():
                                batch_img_dir = gr.Textbox(
                                    label="Ou saisir le chemin d'un dossier local d'images",
                                    placeholder="Ex: C:\\utilisateurs\\images_lots"
                                )
                                batch_img_scan_btn = gr.Button("🔍 Scanner le dossier", variant="secondary")
                            batch_img_status = gr.Markdown("*Aucune image dans le lot.*")
                            batch_img_clear = gr.Button("🗑️ Vider le lot d'images", variant="secondary")

                        with gr.Column(scale=1):
                            gr.Markdown("#### 📄 Fichiers CSV de labels")
                            batch_csv_upload = gr.File(
                                label="Ajouter des CSV de labels (cumulatif)",
                                file_count="multiple",
                                file_types=[".csv"],
                                elem_classes=["horizontal-files"]
                            )
                            batch_csv_status = gr.Markdown("*Aucun fichier CSV chargé.*")
                            batch_csv_clear = gr.Button("🗑️ Vider les CSV", variant="secondary")

                    with gr.Row():
                        batch_zip_images = gr.Checkbox(
                            label="Inclure les images physiques dans le ZIP de sortie",
                            value=False
                        )

                    batch_run_btn = gr.Button("🚀 Fusionner et Vérifier", variant="primary")

                    with gr.Row():
                        batch_log = gr.Textbox(
                            label="📋 Rapport de fusion & Logs",
                            lines=12,
                            interactive=False,
                            placeholder="Le rapport de fusion apparaîtra ici…"
                        )
                    with gr.Row():
                        batch_download = gr.File(label="⬇️ Télécharger le ZIP des résultats", visible=False)
                        batch_df = gr.Dataframe(label="📊 Résultats de la fusion", visible=False)

                    # Accumulation functions
                    def add_images(new_files, current_state):
                        if not new_files:
                            return current_state, f"**Images accumulées :** {len(current_state)}"
                        
                        # Add only new paths avoiding duplicate basenames
                        existing_names = {Path(p).name for p in current_state}
                        for f in new_files:
                            path_str = _get_path(f)
                            name = Path(path_str).name
                            if name not in existing_names:
                                current_state.append(path_str)
                                existing_names.add(name)
                        
                        return current_state, f"**Images accumulées :** {len(current_state)}"

                    def scan_directory(dir_path, current_state):
                        if not dir_path or not os.path.exists(dir_path):
                            return current_state, f"⚠️ Le dossier '{dir_path}' n'existe pas ou le chemin est invalide."
                        
                        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                        found_files = []
                        
                        try:
                            # Fast scanning using os.scandir
                            with os.scandir(dir_path) as it:
                                for entry in it:
                                    if entry.is_file():
                                        ext = Path(entry.name).suffix.lower()
                                        if ext in valid_exts:
                                            found_files.append(entry.path)
                        except Exception as e:
                            return current_state, f"❌ Erreur lors du scan : {e}"
                            
                        if not found_files:
                            return current_state, f"⚠️ Aucune image valide trouvée dans '{dir_path}'."
                            
                        # Add to state avoiding duplicates
                        existing_names = {Path(p).name for p in current_state}
                        added_count = 0
                        for p in found_files:
                            name = Path(p).name
                            if name not in existing_names:
                                current_state.append(p)
                                existing_names.add(name)
                                added_count += 1
                                
                        return current_state, f"**Images accumulées :** {len(current_state)} (+{added_count} ajoutées depuis le dossier)"

                    def add_csvs(new_files, current_state):
                        if not new_files:
                            return current_state, f"**Fichiers CSV accumulés :** {len(current_state)}"
                        
                        existing_paths = set(current_state)
                        for f in new_files:
                            path_str = _get_path(f)
                            if path_str not in existing_paths:
                                current_state.append(path_str)
                                existing_paths.add(path_str)
                                
                        return current_state, f"**Fichiers CSV accumulés :** {len(current_state)}"

                    def clear_images():
                        return [], "*Aucune image dans le lot.*"

                    def clear_csvs():
                        return [], "*Aucun fichier CSV chargé.*"

                    # Connections
                    batch_img_upload.change(
                        fn=add_images,
                        inputs=[batch_img_upload, batch_img_state],
                        outputs=[batch_img_state, batch_img_status]
                    )
                    batch_img_scan_btn.click(
                        fn=scan_directory,
                        inputs=[batch_img_dir, batch_img_state],
                        outputs=[batch_img_state, batch_img_status]
                    )
                    batch_csv_upload.change(
                        fn=add_csvs,
                        inputs=[batch_csv_upload, batch_csv_state],
                        outputs=[batch_csv_state, batch_csv_status]
                    )

                    batch_img_clear.click(
                        fn=clear_images,
                        inputs=[],
                        outputs=[batch_img_state, batch_img_status]
                    )
                    batch_csv_clear.click(
                        fn=clear_csvs,
                        inputs=[],
                        outputs=[batch_csv_state, batch_csv_status]
                    )

                    def fn_merge(img_list, csv_list, zip_imgs):
                        logs_str, zip_p, df_out = run_merge_labels(img_list, csv_list, zip_imgs)
                        return (
                            logs_str,
                            gr.update(value=zip_p, visible=zip_p is not None),
                            gr.update(value=df_out, visible=df_out is not None)
                        )

                    batch_run_btn.click(
                        fn=fn_merge,
                        inputs=[batch_img_state, batch_csv_state, batch_zip_images],
                        outputs=[batch_log, batch_download, batch_df]
                    )


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
