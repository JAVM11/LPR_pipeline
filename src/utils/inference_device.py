"""Selección de dispositivo compartida por YOLO y OCR, sólo al inicializar."""
import os
import torch


def resolve_device(requested="auto"):
    # La variable explícita prevalece sobre defaults históricos de las entradas CLI.
    mode = os.environ.get("INFERENCE_DEVICE", requested).strip().lower()
    if mode not in {"auto", "cuda", "cpu", "cuda:0", "0"}:
        raise ValueError("INFERENCE_DEVICE debe ser auto, cuda o cpu")
    available = torch.cuda.is_available()
    if mode in {"cuda", "cuda:0", "0"} and not available:
        raise RuntimeError("CUDA solicitada pero no disponible. Revise NVIDIA/driver y el override GPU, o use INFERENCE_DEVICE=cpu")
    selected = "cuda:0" if mode != "cpu" and available else "cpu"
    print("Inference device: " + ("CUDA" if selected.startswith("cuda") else "CPU"))
    if selected.startswith("cuda"):
        print("GPU: " + torch.cuda.get_device_name(0))
    elif mode == "auto":
        print("CUDA unavailable - using CPU fallback")
    return selected
