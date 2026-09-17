"""
Pipeline principal de License Plate Recognition (LPR).

Integra:
- YOLOv8 (Ultralytics) para detección de vehículos y placas
- Fast-Plate-OCR para reconocimiento de caracteres
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
import math

from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.ocr.plate_ocr import PlateOCR
from src.tracking.vehicle_tracker import VehicleTracker

from src.utils.validation import validate_confidence, validate_frame, clip_bbox
from src.utils.inference_device import resolve_device
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Resultado estructurado de una detección LPR."""
    track_id: Optional[int]
    vehicle_type: str
    vehicle_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    plate_bbox: Optional[Tuple[int, int, int, int]]
    plate_text: Optional[str]
    plate_confidence: float
    frame_id: int


class LPRPipeline:
    """
    Pipeline completo de reconocimiento de placas.

    Flujo:
    1. Detectar vehículos en el frame (YOLOv8 COCO)
    2. Detectar placa dentro de cada vehículo (YOLOv8 custom)
    3. OCR sobre la placa recortada (Fast-Plate-OCR)
    4. Tracking para mantener IDs consistentes
    """

    def __init__(
        self,
        vehicle_model: str = "yolov8n.pt",
        plate_model: str = "yolov8n.pt",  # Idealmente usar modelo fine-tuned en placas
        ocr_model: str = "cct-s-v2-global-model",
        confidence_threshold: float = 0.5,
        device: str = "auto"  # INFERENCE_DEVICE: auto, cuda o cpu
    ):
        """
        Args:
            vehicle_model: Ruta a pesos YOLOv8 para vehículos (o nombre del modelo hub)
            plate_model: Ruta a pesos YOLOv8 para placas (fine-tuned recomendado)
            ocr_model: Nombre del modelo en Fast-Plate-OCR hub
            confidence_threshold: Confianza mínima para considerar detecciones
            device: Dispositivo de inferencia ("cpu", "cuda:0", etc.)
        """
        validate_confidence(confidence_threshold)
        self.conf_threshold = confidence_threshold
        self.device = resolve_device(device)

        # Inicializar componentes
        logger.info("Inicializando pipeline LPR...")
        self.vehicle_detector = VehicleDetector(vehicle_model, self.device)
        self.plate_detector = PlateDetector(plate_model, self.device)
        self.ocr = PlateOCR(ocr_model, device=self.device)
        self.tracker = VehicleTracker()

        logger.info("Pipeline listo")

    def process_image(self, image_path: str) -> List[DetectionResult]:
        """
        Procesa una imagen estática.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Lista de DetectionResult con placas encontradas
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")

        self.tracker.reset()
        return self._process_frame(img, frame_id=0)

    def process_video(
        self, 
        source: str, 
        output_path: Optional[str] = None,
        show: bool = False
    ) -> List[DetectionResult]:
        """
        Procesa video archivo o stream RTSP.

        Args:
            source: Ruta a video o URL RTSP
            output_path: Si se especifica, guarda video con anotaciones
            show: Mostrar ventana en tiempo real

        Returns:
            Lista consolidada de detecciones únicas
        """
        if output_path and isinstance(source, (str, Path)) and "://" not in str(source):
            src_path, dst_path = Path(source).resolve(), Path(output_path).resolve()
            if src_path == dst_path or (dst_path.exists() and src_path.exists() and src_path.samefile(dst_path)):
                raise ValueError("La salida no puede sobrescribir el video de entrada")

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        writer = None
        best_by_track = {}
        frame_id = 0
        try:
            if not cap.isOpened():
                # No incluir la URL: podría contener credenciales o tokens.
                raise ValueError("No se pudo abrir la fuente de video")
            self.tracker.reset()
            if output_path:
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("FPS de la fuente inválido para guardar video")
                dimensions = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
                if any(not math.isfinite(v) or v < 1 for v in dimensions):
                    raise ValueError("Dimensiones de video inválidas")
                w, h = map(int, dimensions)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise ValueError("No se pudo abrir el archivo de video de salida")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                validate_frame(frame)
                detections = self._process_frame(frame, frame_id)
                # Conservar sólo la mejor lectura por ID, con el mismo criterio de desempate.
                for det in detections:
                    if det.plate_text is not None:
                        previous = best_by_track.get(det.track_id)
                        if previous is None or det.plate_confidence > previous.plate_confidence:
                            best_by_track[det.track_id] = det
                if writer is not None or show:
                    vis_frame = self._draw_results(frame.copy(), detections)
                    if writer is not None:
                        if (frame.shape[1], frame.shape[0]) != (w, h):
                            raise ValueError("Las dimensiones del frame cambiaron durante la grabación")
                        writer.write(vis_frame)
                    if show:
                        cv2.imshow("LPR Pipeline", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                frame_id += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
        return list(best_by_track.values())

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> List[DetectionResult]:
        """Procesa un frame individual."""
        validate_frame(frame)
        results = []

        # 1. Detección de vehículos
        vehicles = self.vehicle_detector.detect(frame, self.conf_threshold)

        # 2. Tracking (asignar IDs)
        tracked_vehicles = self.tracker.update(vehicles, frame_id)

        for vehicle in tracked_vehicles:
            vehicle_bbox = clip_bbox(vehicle.bbox, frame.shape)
            if vehicle_bbox is None:
                continue
            x1, y1, x2, y2 = vehicle_bbox

            # Recortar región del vehículo
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            # 3. Detección de placa dentro del vehículo
            plate_det = self.plate_detector.detect(vehicle_roi, self.conf_threshold)

            plate_bbox = clip_bbox(plate_det.bbox, vehicle_roi.shape) if plate_det is not None else None
            if plate_bbox is None:
                results.append(DetectionResult(
                    track_id=vehicle.track_id,
                    vehicle_type=vehicle.class_name,
                    vehicle_bbox=vehicle_bbox,
                    plate_bbox=None,
                    plate_text=None,
                    plate_confidence=0.0,
                    frame_id=frame_id
                ))
                continue

            # 4. OCR sobre la placa
            px1, py1, px2, py2 = plate_bbox
            plate_roi = vehicle_roi[py1:py2, px1:px2]

            plate_text, plate_conf = self.ocr.recognize(plate_roi)

            # Coordenadas globales de la placa
            global_plate_bbox = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)

            results.append(DetectionResult(
                track_id=vehicle.track_id,
                vehicle_type=vehicle.class_name,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=global_plate_bbox,
                plate_text=plate_text,
                plate_confidence=plate_conf,
                frame_id=frame_id
            ))

        return results

    def _draw_results(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Dibuja anotaciones en el frame."""
        for det in detections:
            # Dibujar vehículo
            x1, y1, x2, y2 = det.vehicle_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"ID:{det.track_id} {det.vehicle_type}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Dibujar placa si existe
            if det.plate_bbox and det.plate_text:
                px1, py1, px2, py2 = det.plate_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

                plate_label = f"{det.plate_text} ({det.plate_confidence:.2f})"
                cv2.putText(frame, plate_label, (px1, py1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def _consolidate_results(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Consolida resultados por track_id, quedándose con la mejor confianza."""
        best_by_track = {}

        for det in detections:
            if det.plate_text is None:
                continue

            if det.track_id not in best_by_track:
                best_by_track[det.track_id] = det
            else:
                if det.plate_confidence > best_by_track[det.track_id].plate_confidence:
                    best_by_track[det.track_id] = det

        return list(best_by_track.values())
