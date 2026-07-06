import re

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
in_detecteur = False
for i, line in enumerate(lines):
    if "with gr.Tab(\"Détecter et découper\"):" in line and "with gr.Tab(\"Détecteur d'identifiant\"):" in "".join(lines[i-5:i]):
        new_lines.append('                with gr.Tab("CARTE"):\n')
        new_lines.append('                    with gr.Tabs():\n')
        new_lines.append('                        ' + line.lstrip())
        in_detecteur = True
        continue
    
    if in_detecteur:
        if "with gr.Tab(\"Get labels images\"):" in line:
            new_lines.append('                with gr.Tab("OM"):\n')
            new_lines.append('                    with gr.Tabs():\n')
            new_lines.append('                        with gr.Tab("retrouver images de labels"):\n')
            continue
            
        if line.strip() == "" and i > 1980: # Just to catch empty lines properly without indenting them infinitely
            # but we know it stops before Tab 4
            pass
            
        if "        # ── Tab 4" in line:
            in_detecteur = False
            new_lines.append(line)
            continue
            
        if line.strip() == "":
            new_lines.append(line)
        else:
            new_lines.append("        " + line)
    else:
        new_lines.append(line)

with open("d:\\psi_work\\ai_project\\momo_ai\\DataToolBox\\app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Modification done!")
