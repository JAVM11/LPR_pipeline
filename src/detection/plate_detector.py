"""
Detección de placas vehiculares usando YOLOv8.

Nota: Para mejores resultados, usar un modelo fine-tuned específicamente 
en placas (ver docs/training.md). Se puede usar el modelo base pero con 
menor precisión.

Crédito: Framework YOLOv8 de Ultralytics.
"""

import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO
from src.utils.validation import (
    validate_confidence, validate_device, validate_frame, validate_model_path, clip_bbox,
)


@dataclass
class PlateDetection:
    """Detección de placa."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (relativo al ROI padre)
    confidence: float


class PlateDetector:
    """
    Detector de placas basado en YOLOv8.

    Idealmente usar un modelo entrenado en dataset de placas como CCPD
    o Roboflow Universe.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Args:
            model_path: Ruta a modelo YOLOv8 fine-tuned en placas
            device: Dispositivo de inferencia
        """
        validate_device(device)
        validate_model_path(model_path)
        self.device = device
        self.model = YOLO(model_path)
        if str(model_path) == "yolov8n.pt":
            logging.getLogger(__name__).warning(
                "yolov8n.pt es COCO, no un detector de placas; configure pesos entrenados para placas"
            )

    def detect(self, vehicle_roi: np.ndarray, conf_threshold: float = 0.5) -> Optional[PlateDetection]:
        """
        Detecta placa en el ROI de un vehículo.

        Args:
            vehicle_roi: Imagen recortada del vehículo (BGR)
            conf_threshold: Confianza mínima

        Returns:
            PlateDetection o None si no se encuentra placa
        """
        if vehicle_roi is None or vehicle_roi.size == 0:
            return None

        validate_frame(vehicle_roi)
        validate_confidence(conf_threshold)
        results = self.model(vehicle_roi, verbose=False, conf=conf_threshold, device=self.device, classes=[0])[0]

        # Tomar la detección con mayor confianza
        best_det = None
        best_conf = 0.0

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                bbox = clip_bbox(box.xyxy[0].tolist(), vehicle_roi.shape)
                if bbox is None:
                    continue
                best_det = PlateDetection(
                    bbox=bbox,
                    confidence=conf
                )
                best_conf = conf

        return best_det
