#!/usr/bin/env python
"""
Script para detectar placas en una imagen.
"""

import argparse
import logging
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def main():
    parser = argparse.ArgumentParser(description="Detección de placas en imagen")
    parser.add_argument("--input", "-i", required=True, help="Ruta a imagen de entrada")
    parser.add_argument("--output", "-o", default="results", help="Directorio de salida")
    parser.add_argument("--vehicle-model", default="yolov8n.pt", help="Modelo YOLO para vehículos")
    parser.add_argument("--plate-model", default="yolov8n.pt", help="Modelo YOLO para placas")
    parser.add_argument("--ocr-model", default="cct-s-v2-global-model", help="Modelo Fast-Plate-OCR")
    parser.add_argument("--conf", type=float, default=0.5, help="Confianza mínima")
    parser.add_argument("--device", default="cpu", help="Dispositivo (cpu, cuda:0)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not 0 <= args.conf <= 1:
        parser.error("--conf debe estar entre 0 y 1")

    if not Path(args.input).is_file():
        parser.error("No existe la imagen de entrada")
    import cv2
    img = cv2.imread(args.input)
    if img is None:
        parser.error("No se pudo decodificar la imagen de entrada")
    from src.pipeline import LPRPipeline

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar pipeline
    print("Inicializando pipeline...")
    lpr = LPRPipeline(
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        ocr_model=args.ocr_model,
        confidence_threshold=args.conf,
        device=args.device
    )

    # Procesar
    print(f"Procesando: {args.input}")
    results = lpr.process_image(args.input)

    # Guardar resultados
    import json

    # Dibujar y guardar imagen anotada
    vis = lpr._draw_results(img, results)
    output_path = output_dir / f"result_{Path(args.input).name}"
    if not cv2.imwrite(str(output_path), vis):
        raise OSError(f"No se pudo guardar la imagen: {output_path}")

    # Guardar JSON
    json_results = []
    for det in results:
        json_results.append({
            "track_id": det.track_id,
            "vehicle_type": det.vehicle_type,
            "vehicle_bbox": det.vehicle_bbox,
            "plate_bbox": det.plate_bbox,
            "plate_text": det.plate_text,
            "plate_confidence": det.plate_confidence
        })

    json_path = output_dir / f"result_{Path(args.input).stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    # Imprimir resumen
    print(f"\nDetecciones encontradas: {len(results)}")
    for det in results:
        if det.plate_text:
            print(f"   🚗 {det.vehicle_type} (ID:{det.track_id}) → Placa: {det.plate_text} ({det.plate_confidence:.2f})")

    print(f"\nResultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
