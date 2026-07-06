import re

backend_code = """def run_om_obb_mode(
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
        labels_dir = out_dir / "labels"
        labels_dir.mkdir()

        for fpath in uploaded_files:
            fpath = _get_path(fpath)
            fname = Path(fpath).name
            stem = Path(fname).stem
            
            log(f"🖼️ Traitement : {fname}")
            try:
                # inference
                results = model.predict(source=fpath, conf=conf, save=False, verbose=False)
                r = results[0]
                
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
        build_zip_to_file(labels_dir, zip_path)
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
                            
                            om_obb_images_state = gr.State([])
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    om_obb_files = gr.File(label="Upload Images (cumulatif)", file_count="multiple", file_types=["image"], elem_classes=["horizontal-files"])
                                    with gr.Row():
                                        om_obb_img_dir = gr.Textbox(label="Ou dossier d'images local", placeholder="Ex: C:\\images")
                                        om_obb_img_scan_btn = gr.Button("🔍 Scanner images", variant="secondary")
                                    with gr.Row():
                                        om_obb_images_status = gr.Markdown("**Images accumulées :** 0")
                                        om_obb_images_clear = gr.Button("🗑️ Vider les images", variant="secondary")
                                with gr.Column(scale=1):
                                    om_obb_conf = gr.Slider(minimum=10, maximum=100, value=25, step=5, label="Seuil de confiance (%)")

                            om_obb_btn = gr.Button("Générer Dataset OBB OM", variant="primary")
                            with gr.Row():
                                om_obb_log = gr.Textbox(label="📋 Logs", lines=8, interactive=False)
                            with gr.Row():
                                om_obb_dl = gr.File(label="⬇️ Télécharger ZIP (labels OBB uniquement)")
                                om_obb_df = gr.Dataframe(label="📊 Résultats")

                            def add_om_obb_images(new_files, current_state):
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

                            def scan_om_obb_img_dir(dir_path, current_state):
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

                                yield from add_om_obb_images(found_files, current_state)

                            om_obb_files.change(
                                fn=add_om_obb_images,
                                inputs=[om_obb_files, om_obb_images_state],
                                outputs=[om_obb_images_state, om_obb_images_status]
                            )
                            om_obb_img_scan_btn.click(
                                fn=scan_om_obb_img_dir,
                                inputs=[om_obb_img_dir, om_obb_images_state],
                                outputs=[om_obb_images_state, om_obb_images_status]
                            )
                            om_obb_images_clear.click(
                                fn=lambda: ([], "**Images accumulées :** 0"),
                                inputs=[],
                                outputs=[om_obb_images_state, om_obb_images_status]
                            )

                            def fn_om_obb(files, conf):
                                logs, zip_p, df = run_om_obb_mode(files, conf)
                                return logs, zip_p, df_update_with_count(df)
                            om_obb_btn.click(fn_om_obb, inputs=[om_obb_images_state, om_obb_conf], outputs=[om_obb_log, om_obb_dl, om_obb_df])
"""

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_backend = False
in_ui = False

for i, line in enumerate(lines):
    # Detect backend block
    if line.startswith("def run_om_obb_mode("):
        in_backend = True
        new_lines.extend(backend_code.splitlines(True))
        continue
    if in_backend:
        if line.startswith("def run_deskew("):
            in_backend = False
            new_lines.append(line)
        continue

    # Detect UI block
    if 'with gr.Tab("Générer labels OBB"):' in line:
        in_ui = True
        new_lines.extend(ui_code.splitlines(True))
        continue
    if in_ui:
        if '        # ── Tab 4 : Reconnaissance (OCR) ─────────────────────────────────────────────' in line:
            in_ui = False
            new_lines.append(line)
        continue
    
    new_lines.append(line)

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Insertion done!")
