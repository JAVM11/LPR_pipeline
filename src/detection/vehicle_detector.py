"""
Detección de vehículos usando YOLOv8 (Ultralytics).

Crédito: Este módulo usa el framework YOLOv8 de Ultralytics.
Repositorio: https://github.com/ultralytics/ultralytics
Licencia: AGPL-3.0
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from ultralytics import YOLO
from src.utils.validation import (
    validate_confidence, validate_device, validate_frame, validate_model_path, clip_bbox,
)


@dataclass
class VehicleDetection:
    """Detección de vehículo."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


class VehicleDetector:
    """
    Detector de vehículos basado en YOLOv8.

    Detecta clases COCO: car (2), motorcycle (3), bus (5), truck (7)
    """

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Args:
            model_path: Ruta a modelo YOLOv8 o nombre (ej. "yolov8n.pt")
            device: "cpu", "cuda:0", etc.
        """
        validate_device(device)
        validate_model_path(model_path)
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[VehicleDetection]:
        """
        Detecta vehículos en un frame.

        Args:
            frame: Imagen BGR (numpy array)
            conf_threshold: Confianza mínima

        Returns:
            Lista de VehicleDetection
        """
        validate_frame(frame)
        validate_confidence(conf_threshold)
        results = self.model(frame, verbose=False, conf=conf_threshold, device=self.device)[0]

        detections = []
        for box in results.boxes:
            bbox = clip_bbox(box.xyxy[0].tolist(), frame.shape)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if bbox is None or cls_id not in self.VEHICLE_CLASSES:
                continue

            detections.append(VehicleDetection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=self.VEHICLE_CLASSES[cls_id]
            ))

        return detections
