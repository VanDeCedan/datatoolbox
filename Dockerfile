# ─────────────────────────────────────────────────────────────
#  Data Toolbox App — Dockerfile (Gradio version)
#  CPU-only, no NVIDIA drivers required.
#  Build : docker build -t data-toolbox-gradio .
#  Run   : docker run -p 7860:7860 data-toolbox-gradio
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="Momo AI"
LABEL description="Data Toolbox — YOLOv8 + Custom CNN Deskewer + Gradio"

# ── Global pip settings ──────────────────────────────────────
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=10
ENV PIP_NO_CACHE_DIR=1

# ── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Install Python deps ──────────────────────────────────────
RUN pip install --upgrade pip

# CPU-only PyTorch first (avoids CUDA downloads)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Copy application code ────────────────────────────────────
RUN mkdir -p weights

# Copy all python source files
COPY app.py detector.py deskewer.py pdf_utils.py ./

# ── Gradio configuration ─────────────────────────────────────
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ── Expose Gradio port ───────────────────────────────────────
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

# ── Entrypoint ───────────────────────────────────────────────
CMD ["python", "app.py"]
