import sys

def main():
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace early return for NO files in all 4 functions
    content = content.replace(
        'if not uploaded_files:\n        return "⚠️ Aucun fichier uploadé.", None, None',
        'if not uploaded_files:\n        yield "⚠️ Aucun fichier uploadé.", None, None\n        return'
    )
    content = content.replace(
        'if mode == "crop_resize" and not card_type:\n        return "⚠️ Erreur: Le type de carte est obligatoire pour lancer le découpage.", None, None',
        'if mode == "crop_resize" and not card_type:\n        yield "⚠️ Erreur: Le type de carte est obligatoire pour lancer le découpage.", None, None\n        return'
    )

    # 2. Replace loop start
    content = content.replace(
        'for uf in uploaded_files:\n            fpath = _get_path(uf)',
        'BATCH_SIZE = 500\n        for idx, uf in enumerate(uploaded_files):\n            if idx % BATCH_SIZE == 0:\n                log(f"🔄 Traitement du lot {idx//BATCH_SIZE + 1}/{(len(uploaded_files) + BATCH_SIZE - 1)//BATCH_SIZE}...")\n            fpath = _get_path(uf)'
    )

    # 3. Replace loop end for classify
    content = content.replace(
        '                    all_skipped.append({"Fichier": fname, "Page": "—", "Raison": "error"})\n\n        if not all_records:',
        '                    all_skipped.append({"Fichier": fname, "Page": "—", "Raison": "error"})\n            \n            if (idx + 1) % BATCH_SIZE == 0:\n                import gc; gc.collect()\n                yield "\\n".join(logs), None, pd.DataFrame(all_records) if all_records else None\n\n        if not all_records:'
    )

    # 4. Replace loop end for deskew
    content = content.replace(
        '                    "Confiance": "—", "Statut": "Erreur",\n                })\n\n        if not all_records:',
        '                    "Confiance": "—", "Statut": "Erreur",\n                })\n            \n            if (idx + 1) % BATCH_SIZE == 0:\n                import gc; gc.collect()\n                yield "\\n".join(logs), None, pd.DataFrame(all_records) if all_records else None\n\n        if not all_records:'
    )

    # 5. Replace loop end for ocr and obb (they share exactly the same code)
    content = content.replace(
        '            except Exception as exc:\n                log(f"  ❌ Erreur : {exc}")\n\n        if not all_records:\n            log("⚠️ Aucun résultat.")\n            return "\\n".join(logs), None, None',
        '            except Exception as exc:\n                log(f"  ❌ Erreur : {exc}")\n            \n            if (idx + 1) % BATCH_SIZE == 0:\n                import gc; gc.collect()\n                yield "\\n".join(logs), None, pd.DataFrame(all_records) if all_records else None\n\n        if not all_records:\n            log("⚠️ Aucun résultat.")\n            yield "\\n".join(logs), None, None\n            return'
    )

    # 7. Replace final returns
    content = content.replace('        return "\\n".join(logs), str(zip_path), df', '        yield "\\n".join(logs), str(zip_path), df')
    content = content.replace('        return "\\n".join(logs), str(csv_path), df', '        yield "\\n".join(logs), str(csv_path), df')
    
    # 8. Replace empty final returns
    content = content.replace(
        '        if not all_records:\n            log("⚠️ Aucun résultat détecté.")\n            return "\\n".join(logs), None, pd.DataFrame(all_skipped) if all_skipped else None',
        '        if not all_records:\n            log("⚠️ Aucun résultat détecté.")\n            yield "\\n".join(logs), None, pd.DataFrame(all_skipped) if all_skipped else None\n            return'
    )
    content = content.replace(
        '        if not all_records:\n            return "\\n".join(logs), None, None',
        '        if not all_records:\n            yield "\\n".join(logs), None, None\n            return'
    )

    # 9. Replace error returns
    content = content.replace('        return f"❌ Erreur inattendue : {exc}", None, None', '        yield f"❌ Erreur inattendue : {exc}", None, None\n        return')
    content = content.replace('        return f"❌ Erreur de chargement du modèle: {e}", None, None', '        yield f"❌ Erreur de chargement du modèle: {e}", None, None\n        return')

    # Now update Gradio endpoints to iterate
    # fn_c1
    content = content.replace(
        '                        logs, zip_p, df = run_classify_mode(files, conf, "classification_only")\n                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)',
        '                        for logs, zip_p, df in run_classify_mode(files, conf, "classification_only"):\n                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)'
    )
    # fn_c2
    content = content.replace(
        '                        logs, zip_p, df = run_classify_mode(files, conf, "crop_and_classify")\n                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)',
        '                        for logs, zip_p, df in run_classify_mode(files, conf, "crop_and_classify"):\n                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)'
    )
    # fn_c3
    content = content.replace(
        '                        logs, zip_p, df = run_classify_mode(files, conf, "yolo_annotation")\n                        return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)',
        '                        for logs, zip_p, df in run_classify_mode(files, conf, "yolo_annotation"):\n                            yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)'
    )
    # deskew
    content = content.replace(
        '                logs, zip_path, df = run_deskew(files, conf_deskew)\n                return (\n                    logs,\n                    gr.update(value=zip_path, visible=zip_path is not None),\n                    df_update_with_count(df, "📊 Résultats de redressement"),\n                )',
        '                for logs, zip_path, df in run_deskew(files, conf_deskew):\n                    yield (\n                        logs,\n                        gr.update(value=zip_path, visible=zip_path is not None),\n                        df_update_with_count(df, "📊 Résultats de redressement"),\n                    )'
    )
    # obb_c1
    content = content.replace(
        '                        logs, zip_p, df = run_obb_mode(files, conf, "crop_resize", card_type=card_type)\n                        return logs, zip_p, df_update_with_count(df)',
        '                        for logs, zip_p, df in run_obb_mode(files, conf, "crop_resize", card_type=card_type):\n                            yield logs, zip_p, df_update_with_count(df)'
    )
    # obb_c2
    content = content.replace(
        '                        logs, zip_p, df = run_obb_mode(files, conf, "generate_labels", int(w), int(h), "")\n                        return logs, zip_p, df_update_with_count(df)',
        '                        for logs, zip_p, df in run_obb_mode(files, conf, "generate_labels", int(w), int(h), ""):\n                            yield logs, zip_p, df_update_with_count(df)'
    )
    # ocr
    content = content.replace(
        '                logs, zip_p, df = run_ocr_mode(files, conf)\n                return logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)',
        '                for logs, zip_p, df in run_ocr_mode(files, conf):\n                    yield logs, gr.update(value=zip_p, visible=zip_p is not None), df_update_with_count(df)'
    )
    
    with open('app_new.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    main()
