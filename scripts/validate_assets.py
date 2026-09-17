"""Validación real reproducible: pesos locales, evidencias de ROI y resultados por imagen."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import sys
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--vehicle-model", required=True)
    parser.add_argument("--plate-model", required=True)
    parser.add_argument("--ocr-model", default="cct-s-v2-global-model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / ".cache" / "ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cv2
    import numpy as np
    from fast_plate_ocr.inference import hub
    # Aislar sólo la ubicación de caché; los pesos y la inferencia son los oficiales.
    hub.MODEL_CACHE_DIR = output / ".cache" / "ocr"
    from src.pipeline import LPRPipeline

    def save_image(path, img):
        if not cv2.imwrite(str(path), img):
            raise OSError(f"No se pudo guardar evidencia: {path}")

    lpr = LPRPipeline(vehicle_model=args.vehicle_model, plate_model=args.plate_model,
                      ocr_model=args.ocr_model, confidence_threshold=args.conf, device=args.device)
    metadata = {
        "arguments": vars(args),
        "versions": {n: version(n) for n in ["ultralytics", "fast-plate-ocr", "torch", "torchvision", "numpy", "onnxruntime"]},
        "opencv_runtime": cv2.__version__,
        "models": {
            "vehicle": {"path": str(Path(args.vehicle_model).resolve()), "sha256": sha256(args.vehicle_model), "classes": lpr.vehicle_detector.model.names},
            "plate": {"path": str(Path(args.plate_model).resolve()), "sha256": sha256(args.plate_model), "classes": lpr.plate_detector.model.names, "accepted_classes": [0]},
            "ocr": {"name": args.ocr_model, "providers": lpr.ocr.recognizer.model.get_providers()},
        },
        "images": [],
    }
    vehicle_detect = lpr.vehicle_detector.detect
    plate_detect = lpr.plate_detector.detect
    trace = {}

    def record_vehicles(*a, **kw):
        result = vehicle_detect(*a, **kw)
        trace["vehicles"] = [asdict(d) for d in result]
        return result

    def record_plate(*a, **kw):
        result = plate_detect(*a, **kw)
        trace["plates"].append(asdict(result) if result is not None else None)
        return result

    lpr.vehicle_detector.detect = record_vehicles
    lpr.plate_detector.detect = record_plate
    for index, source in enumerate(args.input, 1):
        source = Path(source).resolve()
        trace.clear()
        trace["plates"] = []
        start = time.perf_counter()
        results = lpr.process_image(str(source))
        elapsed = time.perf_counter() - start
        img = cv2.imread(str(source))
        directory = output / f"image_{index}"
        directory.mkdir(exist_ok=True)
        save_image(directory / "annotated.jpg", lpr._draw_results(img.copy(), results))
        rows = []
        for det in results:
            row = asdict(det)
            x1, y1, x2, y2 = det.vehicle_bbox
            vehicle_roi = img[y1:y2, x1:x2]
            save_image(directory / f"vehicle_{det.track_id}.png", vehicle_roi)
            if det.plate_bbox is not None:
                px1, py1, px2, py2 = det.plate_bbox
                assert x1 <= px1 < px2 <= x2 and y1 <= py1 < py2 <= y2
                crop = img[py1:py2, px1:px2]
                assert np.array_equal(crop, vehicle_roi[py1-y1:py2-y1, px1-x1:px2-x1])
                save_image(directory / f"plate_{det.track_id}.png", crop)
                row["plate_crop_shape"] = list(crop.shape)
                row["plate_inside_vehicle"] = True
            rows.append(row)
        record = {"source": str(source), "sha256": sha256(source), "shape": list(img.shape),
                  "elapsed_seconds": elapsed, "trace": trace.copy(), "results": rows}
        (directory / "results.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        metadata["images"].append(record)
        print(json.dumps({"image": source.name, "results": rows}, ensure_ascii=True), flush=True)
    metadata["ocr_files"] = {str(p.relative_to(output)): sha256(p) for p in (output / ".cache" / "ocr").rglob("*") if p.is_file()}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
