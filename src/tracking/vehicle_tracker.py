"""
Tracking de vehículos usando SORT (Simple Online and Realtime Tracking).

Crédito: Algoritmo SORT de Alex Bewley.
Repositorio: https://github.com/abewley/sort
Licencia: GPL-3.0

Nota: Esta es una implementación simplificada. Para producción, 
considerar usar DeepSORT o ByteTrack.
"""

import numpy as np
from typing import List
from dataclasses import dataclass

# Intentar importar SORT real, si no, usar versión simplificada
try:
    from sort import Sort
    HAS_SORT = True
except ImportError:
    HAS_SORT = False


@dataclass
class TrackedVehicle:
    """Vehículo trackeado con ID consistente."""
    track_id: int
    bbox: tuple
    class_name: str
    confidence: float


class VehicleTracker:
    """
    Tracker de vehículos.

    Usa SORT si está disponible, sino implementación simple de IoU matching.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Args:
            max_age: Frames máximos sin actualizar antes de eliminar track
            min_hits: Detecciones mínimas antes de confirmar track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

        if HAS_SORT:
            self.sort = Sort(max_age=max_age, min_hits=min_hits)
        else:
            self.sort = None
            # Implementación simple fallback
            self.next_id = 0
            self.tracked_objects = {}  # id -> {bbox, age, hits}

    def reset(self):
        """Comienza una fuente independiente conservando los parámetros del tracker."""
        self.__init__(max_age=self.max_age, min_hits=self.min_hits)

    def update(self, detections: list, frame_id: int) -> List[TrackedVehicle]:
        """
        Actualiza tracks con nuevas detecciones.

        Args:
            detections: Lista de VehicleDetection del detector
            frame_id: ID del frame actual

        Returns:
            Lista de TrackedVehicle con IDs asignados
        """
        if HAS_SORT and self.sort is not None:
            # Usar SORT real
            dets = np.array([[*d.bbox, d.confidence] for d in detections]) if detections else np.empty((0, 5))
            trackers = self.sort.update(dets)

            tracked = []
            for t in trackers:
                x1, y1, x2, y2, track_id = map(int, t)
                # Buscar clase original (aproximación por cercanía)
                cls_name = "vehicle"
                for d in detections:
                    if self._iou((x1,y1,x2,y2), d.bbox) > 0.5:
                        cls_name = d.class_name
                        break

                tracked.append(TrackedVehicle(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_name=cls_name,
                    confidence=1.0
                ))
            return tracked

        else:
            # Fallback simple: IoU matching básico
            return self._simple_update(detections)

    def _simple_update(self, detections: list) -> List[TrackedVehicle]:
        """Actualización simple sin dependencias externas."""
        self.frame_count += 1

        # Actualizar edad de tracks existentes
        to_delete = []
        for tid, obj in self.tracked_objects.items():
            obj['age'] += 1
            if obj['age'] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracked_objects[tid]

        # Asignar detecciones a tracks existentes por IoU
        matched = set()
        for det in detections:
            best_iou = 0.3  # Threshold mínimo
            best_tid = None

            for tid, obj in self.tracked_objects.items():
                if tid in matched:
                    continue
                iou = self._iou(det.bbox, obj['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                # Actualizar track existente
                self.tracked_objects[best_tid]['bbox'] = det.bbox
                self.tracked_objects[best_tid]['age'] = 0
                self.tracked_objects[best_tid]['hits'] += 1
                self.tracked_objects[best_tid]['class_name'] = det.class_name
                matched.add(best_tid)
            else:
                # Crear nuevo track
                self.tracked_objects[self.next_id] = {
                    'bbox': det.bbox,
                    'age': 0,
                    'hits': 1,
                    'class_name': det.class_name
                }
                matched.add(self.next_id)
                self.next_id += 1

        # Retornar solo tracks confirmados
        result = []
        for tid in matched:
            obj = self.tracked_objects[tid]
            if obj['hits'] >= self.min_hits or self.frame_count < self.min_hits:
                result.append(TrackedVehicle(
                    track_id=tid,
                    bbox=obj['bbox'],
                    class_name=obj['class_name'],
                    confidence=1.0
                ))

        return result

    @staticmethod
    def _iou(bbox1: tuple, bbox2: tuple) -> float:
        """Calcula Intersection over Union."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
