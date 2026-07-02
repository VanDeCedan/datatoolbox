---
title: Data Toolbox
emoji: 🪪
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# 🪪 Data Toolbox App — Gradio Edition

Une application Gradio qui détecte, recadre et redresse des cartes d'identité depuis des images et des PDFs, et permet d'organiser des milliers de fichiers en lots.

**Modes disponibles :**
- 🔍 **Classify** — Détection et classification de cartes (YOLOv8) avec options recadrage et redressement
- 📐 **Only Deskew** — Redressement automatique d'images pré-recadrées
- 📁 **Group by Batches** — Regroupement de fichiers en lots de taille définie

**Classes de cartes détectées :** `CEDEAO` · `CIP` · `CNI` · `PASSEPORT`

---

## Lancement rapide (local)

### 1. Installer les dépendances

Double-cliquez sur `setup_venv.bat` ou exécutez :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Démarrer l'app

```bash
python app.py
```

Puis ouvrez **http://localhost:7860** dans votre navigateur.

---

## Lancement via Docker

```bash
docker pull sined34/data-toolbox-gradio
```

```bash
docker run -p 7860:7860 sined34/data-toolbox-gradio
```

Puis ouvrez **http://localhost:7860**.
