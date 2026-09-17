"""
OCR de placas usando Fast-Plate-OCR.

Crédito: Esta librería es desarrollada por ankandrew.
Repositorio: https://github.com/ankandrew/fast-plate-ocr
Licencia: MIT

Modelo usado: cct-s-v2-global-model (ver model zoo en su documentación)
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional
from fast_plate_ocr import LicensePlateRecognizer


class PlateOCR:
    """
    Reconocedor de texto en placas usando Fast-Plate-OCR.

    Fast-Plate-OCR es una librería ligera optimizada para placas,
    mucho más rápida que OCRs genéricos como Tesseract para este caso de uso.
    """

    def __init__(self, model_name: str = "cct-s-v2-global-model", device: Optional[str] = None):
        """
        Args:
            model_name: Nombre del modelo en el hub de fast-plate-ocr
                       Opciones comunes:
                       - "cct-s-v2-global-model" (global, recomendado)
                       - "cct-s-v2-eu-model" (Europa)
                       - "cct-s-v2-us-model" (USA)
        """
        import onnxruntime as ort
        from src.utils.inference_device import resolve_device
        selected = resolve_device() if device is None else device
        available = ort.get_available_providers()
        print(f"ONNX available providers: {available}")
        if selected.startswith("cuda"):
            # Cargar las bibliotecas CUDA/cuDNN del mismo entorno que PyTorch.
            ort.preload_dlls()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDA seleccionada, pero ONNX no dispone de CUDAExecutionProvider; instale onnxruntime-gpu compatible")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(f"ONNX requested providers: {providers}")
        self.recognizer = LicensePlateRecognizer(model_name, providers=providers)
        session_providers = self.recognizer.model.get_providers()
        print(f"ONNX active provider: {session_providers[0]} (session: {session_providers})")
        if selected.startswith("cuda") and "CUDAExecutionProvider" not in session_providers:
            raise RuntimeError("La sesión OCR no pudo activar CUDAExecutionProvider; revise CUDA/cuDNN y el driver NVIDIA")
        if selected.startswith("cuda"):
            self.recognizer.model.disable_fallback()

    def recognize(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Reconoce texto de una imagen de placa.

        Args:
            plate_image: Imagen BGR de la placa recortada

        Returns:
            Tupla (texto_placa, confianza). 
            Si no se reconoce nada, retorna (None, 0.0)
        """
        return self.recognize_batch([plate_image])[0]

    def _prepare_image(self, img):
        if (not isinstance(img, np.ndarray) or img.size == 0
                or img.dtype != np.uint8 or img.ndim not in (2, 3)):
            return None
        mode = self.recognizer.config.image_color_mode
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if mode == "rgb" else img
        if img.shape[2] == 1:
            return self._prepare_image(img[:, :, 0])
        if img.shape[2] != 3:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB if mode == "rgb" else cv2.COLOR_BGR2GRAY)

    def _decode(self, raw_text, char_probs):
        if not isinstance(raw_text, str) or char_probs is None:
            return None, 0.0
        # Retirar sólo padding configurado y espacios exteriores, sin inventar caracteres.
        text = raw_text.rstrip(self.recognizer.config.pad_char)
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        text = text.strip()
        probs = np.asarray(char_probs, dtype=float)
        if (len(text) < 4 or probs.ndim != 1 or len(probs) < end):
            return None, 0.0
        probs = probs[start:end]
        if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            return None, 0.0
        return text, float(probs.mean())

    def recognize_batch(self, plate_images: list) -> list:
        """OCR por lote; mantiene orden y un resultado por cada entrada, incluso inválida."""
        results = [(None, 0.0)] * len(plate_images)
        if not plate_images:
            return results
        try:
            valid = []
            indices = []
            for index, img in enumerate(plate_images):
                prepared = self._prepare_image(img)
                if prepared is not None:
                    valid.append(prepared)
                    indices.append(index)
            if not valid:
                return results
            predictions = self.recognizer.run(valid, return_confidence=True)
            # 1.x devuelve (textos, probabilidades); versiones nuevas, PlatePrediction.
            if isinstance(predictions, tuple) and len(predictions) == 2:
                texts, probabilities = predictions
                if len(texts) != len(valid) or len(probabilities) != len(valid):
                    raise ValueError("Número de resultados OCR inesperado")
                decoded = [self._decode(text, probs) for text, probs in zip(texts, probabilities)]
            else:
                if len(predictions) != len(valid):
                    raise ValueError("Número de resultados OCR inesperado")
                decoded = [self._decode(pred.plate, pred.char_probs) for pred in predictions]
            for index, result in zip(indices, decoded):
                results[index] = result
            return results
        except Exception:
            logging.getLogger(__name__).exception("Error durante inferencia OCR")
            return [(None, 0.0)] * len(plate_images)
