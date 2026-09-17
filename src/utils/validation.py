"""Validaciones compartidas sin cargar modelos ni modificar configuración."""
from pathlib import Path
import numpy as np


def validate_confidence(value):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("La confianza debe ser un número finito entre 0 y 1")


def validate_device(device):
    value = str(device).lower()
    if value == "cuda" or value.startswith("cuda:") or value.isdigit():
        import torch
        index = 0 if value == "cuda" else int(value.split(":")[-1])
        if not torch.cuda.is_available() or not 0 <= index < torch.cuda.device_count():
            raise ValueError("GPU CUDA solicitada no disponible; use device='cpu' o una GPU válida")


def validate_model_path(model_path):
    value = str(model_path)
    if not value.strip():
        raise ValueError("El modelo no puede estar vacío")
    # Los nombres del hub sin directorio conservan la descarga de Ultralytics.
    if "://" not in value:
        path = Path(value)
        if (path.parent != Path(".") or path.suffix.lower() in {".onnx", ".engine"}) and not path.is_file():
            raise FileNotFoundError(f"No existe el archivo de modelo: {path}")


def validate_frame(frame):
    if (not isinstance(frame, np.ndarray) or frame.size == 0
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8):
        raise ValueError("La imagen debe ser un array BGR uint8 no vacío (H, W, 3)")


def clip_bbox(bbox, shape):
    coords = np.asarray(bbox, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = coords.astype(int)
    x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None
