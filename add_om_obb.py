import re

backend_code = """
_om_obb_model: Optional[object] = None

def get_om_obb_model():
    global _om_obb_model
    if _om_obb_model is None:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        
        # Download from Nicias/det_om
        model_path = hf_hub_download(repo_id="Nicias/det_om", filename="best.onnx")
        _om_obb_model = YOLO(model_path, task='obb')
    return _om_obb_model

def run_om_obb_mode(
    uploaded_files,
    conf_pct: int,
) -> tuple[str, str | None, object | None]:
    if not uploaded_files:
        return "⚠️ Aucun fichier uploadé.", None, None

    conf = conf_pct / 100.0
    logs = []
    def log(msg: str): logs.append(msg)

    log("🔮 Téléchargement / Chargement du modèle OM OBB (best.onnx)…")
    try:
        model = get_om_obb_model()
    except Exception as e:
        return f"❌ Erreur de chargement du modèle: {e}", None, None

    tmp_root = tempfile.mkdtemp()
    all_records = []
    import cv2
    import numpy as np

    try:
        tmp_path = Path(tmp_root)
        out_dir = tmp_path / "output_om_obb"
        out_dir.mkdir()
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
                # inference
                results = model.predict(source=fpath, conf=conf, save=False, verbose=False)
                r = results[0]
                
                # Copy image
                img_path = images_dir / fname
                import shutil
                shutil.copy2(fpath, img_path)
                
                labels_text = ""
                num_labels = 0
                if hasattr(r, "obb") and r.obb is not None:
                    obb = r.obb
                    if hasattr(obb, "xyxyxyxyn") and obb.xyxyxyxyn is not None and len(obb.xyxyxyxyn) > 0:
                        # get normalized coordinates using obb.xyxyxyxyn
                        boxes = obb.xyxyxyxyn.cpu().numpy() if hasattr(obb.xyxyxyxyn, "cpu") else obb.xyxyxyxyn
                        classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb.cls, "cpu") else obb.cls.astype(int)
                        confs = obb.conf.cpu().numpy() if hasattr(obb.conf, "cpu") else obb.conf

                        for poly, cls_id, conf_val in zip(boxes, classes, confs):
                            if conf_val < conf:
                                continue
                            # format: class x1 y1 x2 y2 x3 y3 x4 y4
                            flat_poly = poly.reshape(-1)
                            poly_str = " ".join([f"{coord:.6f}" for coord in flat_poly])
                            labels_text += f"{int(cls_id)} {poly_str}\\n"
                            num_labels += 1
                            
                lbl_path = labels_dir / f"{stem}.txt"
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write(labels_text)
                    
                log(f"  ✅ Généré {num_labels} labels pour {fname}")
                all_records.append({"Fichier": fname, "Labels": num_labels})

            except Exception as exc:
                log(f"  ❌ Erreur : {exc}")

        if not all_records:
            log("⚠️ Aucun résultat.")
            return "\\n".join(logs), None, None

        log("📦 Génération du ZIP…")
        zip_path = tmp_path / "results_om_obb.zip"
        build_zip_to_file(out_dir, zip_path)
        zip_size = zip_path.stat().st_size
        log(f"✅ ZIP prêt — {zip_size / 1024:.0f} KB")

        import pandas as pd
        df = pd.DataFrame(all_records)
        return "\\n".join(logs), str(zip_path), df

    except Exception as exc:
        try:
            import shutil
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        return f"❌ Erreur inattendue : {exc}", None, None

"""

ui_code = """                        with gr.Tab("Générer labels OBB"):
                            gr.Markdown("### Générer un dataset YOLO OBB avec le modèle OM")
                            with gr.Row():
                                with gr.Column(scale=1):
                                    om_obb_files = gr.File(label="Upload Images", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                                    om_obb_files.change(fn=lambda f: gr.update(label=f"Upload Images ({len(f)} fichiers importés)") if f else gr.update(label="Upload Images"), inputs=om_obb_files, outputs=om_obb_files)
                                with gr.Column(scale=1):
                                    om_obb_conf = gr.Slider(minimum=10, maximum=100, value=25, step=5, label="Seuil de confiance (%)")

                            om_obb_btn = gr.Button("Générer Dataset OBB OM", variant="primary")
                            with gr.Row():
                                om_obb_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                            with gr.Row():
                                om_obb_dl = gr.File(label="⬇️ Télécharger ZIP (images + labels)")
                                om_obb_df = gr.Dataframe(label="📊 Résultats")

                            def fn_om_obb(files, conf):
                                logs, zip_p, df = run_om_obb_mode(files, conf)
                                return logs, zip_p, df_update_with_count(df)
                            om_obb_btn.click(fn_om_obb, inputs=[om_obb_files, om_obb_conf], outputs=[om_obb_log, om_obb_dl, om_obb_df])
"""

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if line.startswith("def run_deskew("):
        new_lines.extend([backend_code, "\n"])
    if "        # ── Tab 4 : Reconnaissance (OCR) ─────────────────────────────────────────────" in line:
        new_lines.append(ui_code + "\n")
    new_lines.append(line)

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Insertion done!")
