#!/usr/bin/env python
"""
Script para detectar placas en video o stream RTSP.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))



def main():
    parser = argparse.ArgumentParser(description="Detección de placas en video/stream")
    parser.add_argument("--input", "-i", required=True, help="Video o RTSP URL")
    parser.add_argument("--output", "-o", help="Video de salida (opcional)")
    parser.add_argument("--show", action="store_true", help="Mostrar en tiempo real")
    parser.add_argument("--vehicle-model", default="yolov8n.pt")
    parser.add_argument("--plate-model", default="yolov8n.pt")
    parser.add_argument("--ocr-model", default="cct-s-v2-global-model")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not 0 <= args.conf <= 1:
        parser.error("--conf debe estar entre 0 y 1")

    if "://" not in args.input and not Path(args.input).is_file():
        parser.error("No existe el video de entrada")
    if args.output and "://" not in args.input:
        src_path, dst_path = Path(args.input).resolve(), Path(args.output).resolve()
        if src_path == dst_path or (dst_path.exists() and src_path.samefile(dst_path)):
            parser.error("La salida no puede sobrescribir el video de entrada")
    from src.pipeline import LPRPipeline

    lpr = LPRPipeline(
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        ocr_model=args.ocr_model,
        confidence_threshold=args.conf,
        device=args.device
    )

    print("Procesando fuente de video...")
    results = lpr.process_video(args.input, args.output, args.show)

    print(f"\n Placas únicas detectadas: {len(results)}")
    for det in results:
        if det.plate_text:
            print(f"   {det.plate_text} (conf: {det.plate_confidence:.2f})")

    # Guardar CSV
    import csv
    csv_path = Path(args.output).parent / "plates.csv" if args.output else "plates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "vehicle_type", "plate_text", "confidence"])
        for det in results:
            writer.writerow([det.track_id, det.vehicle_type, det.plate_text, det.plate_confidence])

    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    main()
