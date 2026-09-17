"""Regresiones offline: modelos simulados, arrays y codecs OpenCV reales."""
import contextlib
import csv
import io
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(os.environ.get("LPR_TEST_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
# Aislar cargas de modelos; nunca instanciar YOLO ni descargar pesos durante pruebas.
saved_modules = {name: sys.modules.get(name) for name in ("ultralytics", "fast_plate_ocr")}
sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=Mock())
sys.modules["fast_plate_ocr"] = types.SimpleNamespace(LicensePlateRecognizer=Mock())
try:
    from src.detection.vehicle_detector import VehicleDetection, VehicleDetector
    from src.detection.plate_detector import PlateDetection, PlateDetector
    from src.ocr.plate_ocr import PlateOCR
    from src.tracking.vehicle_tracker import VehicleTracker, TrackedVehicle
    from src.pipeline import LPRPipeline, DetectionResult
finally:
    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
from src.utils.validation import validate_confidence, validate_device, validate_model_path


def pipeline():
    obj = LPRPipeline.__new__(LPRPipeline)
    obj.conf_threshold = 0.5
    obj.tracker = Mock()
    obj.vehicle_detector = Mock()
    obj.plate_detector = Mock()
    obj.ocr = Mock()
    return obj


def detection(track=1, text="ABC123", conf=0.8, frame=0):
    return DetectionResult(track, "car", (0, 0, 20, 20), (1, 1, 10, 5), text, conf, frame)


def capture(frames=()):
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.side_effect = [(True, x) for x in frames] + [(False, None)]
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 29.97, cv2.CAP_PROP_FRAME_WIDTH: 32,
        cv2.CAP_PROP_FRAME_HEIGHT: 24,
    }[prop]
    return cap


class DetectorTests(unittest.TestCase):
    def test_optional_import(self):
        self.assertIsNone(VehicleDetection((0, 0, 4, 4), .8, 2, "car").track_id)

    def test_onnx_passes_device_at_inference_without_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)/"model.onnx"
            model_path.touch()
            for cls, module in [(VehicleDetector, "vehicle_detector"), (PlateDetector, "plate_detector")]:
                with self.subTest(cls=cls), patch(f"src.detection.{module}.YOLO") as factory:
                    model = factory.return_value
                    model.return_value = [types.SimpleNamespace(boxes=[])]
                    obj = cls(str(model_path), "cpu")
                    obj.detect(np.zeros((8, 8, 3), np.uint8))
                    model.to.assert_not_called()
                    self.assertEqual(model.call_args.kwargs["device"], "cpu")

    def test_boxes_clipped_and_invalid_skipped(self):
        def box(coords, cls=2, conf=.9):
            return types.SimpleNamespace(xyxy=np.array([coords]), cls=[cls], conf=[conf])
        boxes = [box([-5,-5,99,99]), box([3,3,2,2]), box([float("nan"),0,2,2]), box([0,0,5,5],0)]
        obj = VehicleDetector.__new__(VehicleDetector)
        obj.device = "cpu"
        obj.model = Mock(return_value=[types.SimpleNamespace(boxes=boxes)])
        result = obj.detect(np.zeros((10,20,3),np.uint8))
        self.assertEqual([x.bbox for x in result], [(0,0,20,10)])

    def test_plate_skips_invalid_best_box(self):
        obj=PlateDetector.__new__(PlateDetector); obj.device="cpu"
        obj.model=Mock(return_value=[types.SimpleNamespace(boxes=[
            types.SimpleNamespace(xyxy=np.array([[4,4,1,1]]),conf=[.99],cls=[0]),
            types.SimpleNamespace(xyxy=np.array([[-1,-2,9,9]]),conf=[.8],cls=[0]),
        ])])
        self.assertEqual(obj.detect(np.zeros((8,8,3),np.uint8)).bbox,(0,0,8,8))

    def test_plate_only_accepts_class_zero(self):
        obj=PlateDetector.__new__(PlateDetector); obj.device="cpu"
        obj.model=Mock(return_value=[types.SimpleNamespace(boxes=[
            types.SimpleNamespace(xyxy=np.array([[0,0,8,8]]),conf=[.99],cls=[1]),
            types.SimpleNamespace(xyxy=np.array([[1,1,7,5]]),conf=[.8],cls=[0]),
        ])])
        result=obj.detect(np.zeros((8,8,3),np.uint8))
        self.assertEqual(result.bbox,(1,1,7,5))
        self.assertEqual(obj.model.call_args.kwargs["classes"],[0])

    def test_invalid_frame_rejected_before_model(self):
        obj=VehicleDetector.__new__(VehicleDetector); obj.model=Mock()
        for value in [None, np.zeros((0,4,3),np.uint8), np.zeros((4,4)), np.zeros((4,4,4),np.uint8)]:
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                obj.detect(value)
        obj.model.assert_not_called()

    def test_missing_model_and_hub_name(self):
        with self.assertRaises(FileNotFoundError):
            validate_model_path("missing-dir/custom.pt")
        validate_model_path("yolov8n.pt")
        with self.assertRaises(ValueError):
            validate_model_path("")

    def test_bad_confidence(self):
        for value in [-1,1.1,float("nan"),float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_confidence(value)
        validate_confidence(0); validate_confidence(1)

    def test_missing_cuda_fails_explicitly(self):
        torch=types.SimpleNamespace(cuda=Mock())
        torch.cuda.is_available.return_value=False
        with patch.dict(sys.modules,{"torch":torch}), self.assertRaisesRegex(ValueError,"CPU|cpu"):
            validate_device("cuda:0")

    def test_bad_gpu_index(self):
        torch=types.SimpleNamespace(cuda=Mock())
        torch.cuda.is_available.return_value=True
        torch.cuda.device_count.return_value=1
        with patch.dict(sys.modules,{"torch":torch}):
            validate_device("cuda:0")
            with self.assertRaises(ValueError): validate_device("cuda:2")


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.ocr=PlateOCR.__new__(PlateOCR)
        self.ocr.recognizer=Mock()
        self.ocr.recognizer.config=types.SimpleNamespace(pad_char="_",image_color_mode="grayscale")
        self.img=np.full((10,20,3),(255,0,0),np.uint8)

    def test_tuple_contract_and_real_confidence(self):
        self.ocr.recognizer.run.return_value=(["ABC123__"],np.array([[.8]*6+[1.,1.]]))
        text,conf=self.ocr.recognize(self.img)
        self.assertEqual(text,"ABC123"); self.assertAlmostEqual(conf,.8)
        sent=self.ocr.recognizer.run.call_args.args[0][0]
        np.testing.assert_array_equal(sent,cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY))

    def test_modern_prediction_contract(self):
        self.ocr.recognizer.run.return_value=[types.SimpleNamespace(plate="ABC123",char_probs=np.array([.7]*6+[1.,1.]))]
        text,conf=self.ocr.recognize(self.img)
        self.assertEqual(text,"ABC123"); self.assertAlmostEqual(conf,.7)

    def test_rgb_configuration(self):
        self.ocr.recognizer.config.image_color_mode="rgb"
        self.ocr.recognizer.run.return_value=(["ABCD"],np.ones((1,4)))
        self.ocr.recognize(self.img)
        np.testing.assert_array_equal(self.ocr.recognizer.run.call_args.args[0][0],cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))

    def test_batch_alignment(self):
        self.ocr.recognizer.run.return_value=(["ABCD", "WXYZ"],np.ones((2,4)))
        result=self.ocr.recognize_batch([None,self.img,np.zeros((0,2,3),np.uint8),self.img])
        self.assertEqual(result,[(None,0.),("ABCD",1.),(None,0.),("WXYZ",1.)])

    def test_all_invalid_no_inference(self):
        values=[None,np.zeros((0,2,3),np.uint8),np.zeros((2,2,4),np.uint8),"path",np.zeros((2,2,3))]
        self.assertEqual(self.ocr.recognize_batch(values),[(None,0.)]*len(values))
        self.ocr.recognizer.run.assert_not_called()
        self.assertEqual(self.ocr.recognize_batch([]),[])

    def test_short_empty_or_nonfinite_predictions(self):
        for text,probs in [("A___",[1]*4),("____",[1]*4),("ABCD",[float("nan")]*4),("ABCD",[2]*4),("ABCD",None)]:
            with self.subTest(text=text,probs=probs):
                self.ocr.recognizer.run.return_value=[types.SimpleNamespace(plate=text,char_probs=probs)]
                self.assertEqual(self.ocr.recognize(self.img),(None,0.))

    def test_exception_logged_and_batch_shape_preserved(self):
        self.ocr.recognizer.run.side_effect=RuntimeError("inference failed")
        with self.assertLogs("src.ocr.plate_ocr",level="ERROR"):
            self.assertEqual(self.ocr.recognize_batch([self.img,None]),[(None,0.)]*2)

    def test_wrong_result_count(self):
        self.ocr.recognizer.run.return_value=(["ABCD"],np.ones((1,4)))
        with self.assertLogs("src.ocr.plate_ocr",level="ERROR"):
            self.assertEqual(self.ocr.recognize_batch([self.img,self.img]),[(None,0.)]*2)

    def test_installed_ocr_pre_and_postprocessing_without_weights(self):
        if importlib.util.find_spec("fast_plate_ocr") is None:
            self.skipTest("fast-plate-ocr no instalado")
        from fast_plate_ocr import LicensePlateRecognizer
        from fast_plate_ocr.inference import config as ocr_config
        PlateOCRConfig = getattr(ocr_config, "PlateConfig", None) or ocr_config.PlateOCRConfig
        # Usar run() real con backend ONNX simulado; no ejecutar el constructor que descarga pesos.
        real=LicensePlateRecognizer.__new__(LicensePlateRecognizer)
        real.config=PlateOCRConfig(max_plate_slots=6,alphabet="ABCD_",pad_char="_",img_height=8,img_width=16)
        logits=np.zeros((1,6,5),dtype=np.float32)
        for i,c in enumerate("ABCD__"):
            logits[0,i,"ABCD_".index(c)]=.8
        real.has_region_head=False
        real.plate_output_name="plate"
        real.region_output_name=None
        real.model=Mock();real.model.run.return_value=[logits]
        self.ocr.recognizer=real
        text,conf=self.ocr.recognize(self.img)
        self.assertEqual(text,"ABCD");self.assertAlmostEqual(conf,.8)
        actual=real.model.run.call_args.args[1]["input"]
        self.assertEqual(actual.shape,(1,8,16,1))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.mock=patch("src.tracking.vehicle_tracker.HAS_SORT",False)
        self.mock.start();self.addCleanup(self.mock.stop)
        self.tracker=VehicleTracker()

    def test_overlapping_detections_get_distinct_ids(self):
        det=VehicleDetection((0,0,20,20),.9,2,"car")
        first=self.tracker.update([det,det],0)
        self.assertEqual(len(first),2)
        self.assertEqual(len({t.track_id for t in first}),2)
        second=self.tracker.update([det,det],1)
        self.assertEqual({x.track_id for x in second},{x.track_id for x in first})

    def test_track_expiration(self):
        self.tracker.max_age=2
        det=VehicleDetection((0,0,20,20),.9,2,"car")
        self.tracker.update([det],0)
        for i in range(3):self.tracker.update([],i+1)
        self.assertEqual(self.tracker.tracked_objects,{})

    def test_reset_preserves_parameters(self):
        self.tracker.max_age=7;self.tracker.min_hits=2
        self.tracker.update([VehicleDetection((0,0,20,20),.9,2,"car")],0)
        self.tracker.reset()
        self.assertEqual((self.tracker.max_age,self.tracker.min_hits),(7,2))
        self.assertEqual(self.tracker.frame_count,0)
        self.assertEqual(self.tracker.tracked_objects,{})

    def test_sort_branch_reset_and_empty_input(self):
        with patch("src.tracking.vehicle_tracker.HAS_SORT",True), patch("src.tracking.vehicle_tracker.Sort",create=True) as sort:
            sort.return_value.update.return_value=[]
            tracker=VehicleTracker()
            self.assertEqual(tracker.update([],0),[])
            self.assertEqual(sort.return_value.update.call_args.args[0].shape,(0,5))
            tracker.reset();self.assertEqual(sort.call_count,2)


class PipelineTests(unittest.TestCase):
    def test_crop_coordinates_match_pixels(self):
        obj=pipeline();frame=np.zeros((20,30,3),np.uint8)
        obj.tracker.update.return_value=[TrackedVehicle(1,(-5,-4,25,18),"car",.9)]
        obj.plate_detector.detect.return_value=PlateDetection((-2,-1,12,8),.9)
        obj.ocr.recognize.return_value=("ABCD",.8)
        result=obj._process_frame(frame,0)[0]
        self.assertEqual(result.vehicle_bbox,(0,0,25,18))
        self.assertEqual(result.plate_bbox,(0,0,12,8))
        self.assertEqual(obj.ocr.recognize.call_args.args[0].shape,(8,12,3))

    def test_invalid_plate_returns_vehicle_without_ocr(self):
        obj=pipeline();obj.tracker.update.return_value=[TrackedVehicle(1,(0,0,20,20),"car",.9)]
        obj.plate_detector.detect.return_value=PlateDetection((5,5,2,2),.9)
        result=obj._process_frame(np.zeros((24,32,3),np.uint8),0)
        self.assertIsNone(result[0].plate_text);self.assertIsNone(result[0].plate_bbox)
        obj.ocr.recognize.assert_not_called()

    def test_images_reset_state(self):
        obj=pipeline();obj._process_frame=Mock(return_value=[])
        with patch("src.pipeline.cv2.imread",return_value=np.zeros((8,8,3),np.uint8)):
            obj.process_image("a");obj.process_image("b")
        self.assertEqual(obj.tracker.reset.call_count,2)

    def test_missing_image_no_tracking(self):
        obj=pipeline()
        with patch("src.pipeline.cv2.imread",return_value=None),self.assertRaises(ValueError):
            obj.process_image("missing")
        obj.tracker.reset.assert_not_called()

    def test_online_consolidation_matches_existing_rule_and_skips_drawing(self):
        obj=pipeline();frame=np.zeros((24,32,3),np.uint8)
        detections=[[detection(conf=.7),detection(2,None)], [detection(conf=.9,frame=1)], [detection(conf=.9,frame=2),detection(3)]]
        obj._process_frame=Mock(side_effect=detections);obj._draw_results=Mock()
        cap=capture([frame]*3)
        with patch("src.pipeline.cv2.VideoCapture",return_value=cap):
            actual=obj.process_video("input.mp4")
        expected=obj._consolidate_results([d for group in detections for d in group])
        self.assertEqual(actual,expected);obj._draw_results.assert_not_called()
        cap.release.assert_called_once();obj.tracker.reset.assert_called_once()

    def test_fractional_fps_and_parent_directory(self):
        obj=pipeline();cap=capture()
        with tempfile.TemporaryDirectory() as tmp,patch("src.pipeline.cv2.VideoCapture",return_value=cap),patch("src.pipeline.cv2.VideoWriter") as writer:
            path=Path(tmp)/"new"/"out.mp4"
            obj.process_video("in.mp4",str(path))
            self.assertAlmostEqual(writer.call_args.args[2],29.97)
            self.assertTrue(path.parent.is_dir())
            writer.return_value.release.assert_called_once()
        cap.release.assert_called_once()

    def test_bad_fps_releases_capture(self):
        for value in [0,float("nan"),float("inf")]:
            cap=capture();cap.get.side_effect=None;cap.get.return_value=value
            with self.subTest(value=value),patch("src.pipeline.cv2.VideoCapture",return_value=cap),self.assertRaises(ValueError):
                pipeline().process_video("in.mp4","out.mp4")
            cap.release.assert_called_once()

    def test_bad_dimensions_release_capture(self):
        cap=capture();cap.get.side_effect=[30,float("nan"),20]
        with patch("src.pipeline.cv2.VideoCapture",return_value=cap),self.assertRaises(ValueError):
            pipeline().process_video("in.mp4","out.mp4")
        cap.release.assert_called_once()

    def test_failed_writer_releases_everything(self):
        cap=capture()
        with tempfile.TemporaryDirectory() as tmp,patch("src.pipeline.cv2.VideoCapture",return_value=cap),patch("src.pipeline.cv2.VideoWriter") as writer:
            writer.return_value.isOpened.return_value=False
            with self.assertRaises(ValueError):pipeline().process_video("in.mp4",str(Path(tmp)/"out.mp4"))
            writer.return_value.release.assert_called_once()
        cap.release.assert_called_once()

    def test_constructor_error_releases_capture(self):
        cap=capture()
        with tempfile.TemporaryDirectory() as tmp,patch("src.pipeline.cv2.VideoCapture",return_value=cap),patch("src.pipeline.cv2.VideoWriter",side_effect=RuntimeError("codec")),self.assertRaises(RuntimeError):
            pipeline().process_video("in.mp4",str(Path(tmp)/"out.mp4"))
        cap.release.assert_called_once()

    def test_inference_error_releases_video_resources(self):
        cap=capture([np.zeros((24,32,3),np.uint8)]);obj=pipeline()
        obj._process_frame=Mock(side_effect=RuntimeError("infer"))
        with tempfile.TemporaryDirectory() as tmp,patch("src.pipeline.cv2.VideoCapture",return_value=cap),patch("src.pipeline.cv2.VideoWriter") as writer:
            with self.assertRaises(RuntimeError):obj.process_video("in.mp4",str(Path(tmp)/"out.mp4"))
            writer.return_value.release.assert_called_once()
        cap.release.assert_called_once()

    def test_source_error_does_not_expose_credentials(self):
        cap=capture();cap.isOpened.return_value=False
        with patch("src.pipeline.cv2.VideoCapture",return_value=cap),self.assertRaises(ValueError) as error:
            pipeline().process_video("rtsp://admin:secret@example.invalid/live?token=private")
        self.assertNotIn("secret",str(error.exception));self.assertNotIn("private",str(error.exception))
        cap.release.assert_called_once()

    def test_reject_same_input_output_before_open(self):
        with patch("src.pipeline.cv2.VideoCapture") as cap,self.assertRaises(ValueError):
            pipeline().process_video("input.mp4","./input.mp4")
        cap.assert_not_called()

    def test_real_synthetic_video_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.avi";output=Path(tmp)/"nested"/"output.mp4"
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"MJPG"),12.5,(32,24))
            if not writer.isOpened():self.skipTest("Codec MJPG no disponible")
            for i in range(4):writer.write(np.full((24,32,3),i*30,np.uint8))
            writer.release()
            obj=pipeline();obj._process_frame=Mock(side_effect=lambda frame,index:[detection(conf=.5+index*.1,frame=index)])
            result=obj.process_video(str(source),str(output))
            self.assertEqual(result[0].frame_id,3)
            cap=cv2.VideoCapture(str(output))
            try:
                self.assertTrue(cap.isOpened());self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),4)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS),12.5,places=1)
            finally:cap.release()


class CLITests(unittest.TestCase):
    def load_cli(self, name):
        spec=importlib.util.spec_from_file_location("test_cli_"+name,ROOT/"scripts"/name)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_image_json_schema_and_output_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.png";output=Path(tmp)/"results"
            cv2.imwrite(str(source),np.zeros((24,32,3),np.uint8))
            obj=pipeline();obj.process_image=Mock(return_value=[detection()])
            cli=self.load_cli("detect_image.py")
            with patch("src.pipeline.LPRPipeline",return_value=obj),patch.object(sys,"argv",["detect_image","-i",str(source),"-o",str(output)]),contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            rows=json.loads((output/"result_input.json").read_text())
            self.assertEqual(set(rows[0]),{"track_id","vehicle_type","vehicle_bbox","plate_bbox","plate_text","plate_confidence"})
            self.assertEqual(rows[0]["plate_text"],"ABC123")
            self.assertIsNotNone(cv2.imread(str(output/"result_input.png")))

    def test_failed_image_write_reports_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"input.png";output=Path(tmp)/"results"
            cv2.imwrite(str(source),np.zeros((24,32,3),np.uint8))
            obj=pipeline();obj.process_image=Mock(return_value=[])
            cli=self.load_cli("detect_image.py")
            with patch("src.pipeline.LPRPipeline",return_value=obj),patch.object(sys,"argv",["image","-i",str(source),"-o",str(output)]),patch.object(cv2,"imwrite",return_value=False),contextlib.redirect_stdout(io.StringIO()),self.assertRaises(OSError):
                cli.main()
            self.assertFalse((output/"result_input.json").exists())

    def test_video_csv_schema_and_no_url_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=pipeline();obj.process_video=Mock(return_value=[detection()])
            cli=self.load_cli("detect_video.py");output=Path(tmp)/"out.mp4"
            stream=io.StringIO()
            with patch("src.pipeline.LPRPipeline",return_value=obj),patch.object(sys,"argv",["video","-i","rtsp://user:secret@example.invalid/live","-o",str(output)]),contextlib.redirect_stdout(stream):
                cli.main()
            with (Path(tmp)/"plates.csv").open(newline="",encoding="utf-8") as f:
                rows=list(csv.reader(f))
            self.assertEqual(rows[0],["track_id","vehicle_type","plate_text","confidence"])
            self.assertEqual(rows[1],["1","car","ABC123","0.8"])
            self.assertNotIn("secret",stream.getvalue())

    def test_help_without_model_dependencies(self):
        for file in ["detect_image.py","detect_video.py"]:
            result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts"/file),"--help"],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("--plate-model",result.stdout)

    def test_missing_input_checked_before_model_loading(self):
        for file in ["detect_image.py","detect_video.py"]:
            result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts"/file),"-i","__missing_input__"],capture_output=True,text=True)
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertNotIn("ModuleNotFoundError",result.stderr)

    def test_invalid_confidence(self):
        result=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/detect_image.py"),"-i","missing","--conf","nan"],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("--conf",result.stderr)


if __name__ == "__main__":
    unittest.main()
