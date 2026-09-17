# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/lpr-ultralytics \
    MPLCONFIGDIR=/tmp/lpr-matplotlib \
    HOME=/results \
    INFERENCE_DEVICE=auto

# OpenCV es requerido por ambos paquetes; ambas distribuciones usan la misma versión.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/lpr-ultralytics /tmp/lpr-matplotlib

WORKDIR /app
COPY requirements-docker.lock /app/requirements-docker.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && python -m pip install --require-hashes --no-build-isolation -r requirements-docker.lock \
    && python -m pip check \
    && python -c "import torch, onnxruntime as ort; assert torch.version.cuda == '12.8'; assert 'CUDAExecutionProvider' in ort.get_available_providers(); print('CUDA build:', torch.version.cuda, 'ORT:', ort.__version__)"

COPY src/ /app/src/
COPY scripts/ /app/scripts/

# El script existente guarda detecciones, anotaciones y JSON; no se cambia el pipeline.
ENTRYPOINT ["python", "-B", "scripts/detect_image.py", "--vehicle-model", "/models/yolov8n.pt", "--plate-model", "/models/placa2.pt", "--ocr-model", "cct-s-v2-global-model", "--device", "auto", "--conf", "0.5", "--output", "/results"]
CMD ["--input", "/assets/free-photo-of-calle-vehiculo-coche-deportivo-urbano.jpg"]
