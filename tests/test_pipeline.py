"""تست‌های خودکار واحد برای هسته تشخیص پلاک ONNX، نگاشت هندسی و کلاینت API.

این تست‌ها در هر بار اجرای GitHub Actions پیش از ساخت APK اجرا می‌شوند.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

# افزودن پوشه src به مسیر ایمپورت
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AppSettings, CLASS_ID_PLATE, format_iranian_plate
from core.geometry import (
    compute_android_rotation_degrees,
    normalized_box_to_canvas,
    rotate_pil_clockwise,
)
from core.plate_detector import (
    Detection,
    PersianPlateEngine,
    decode_yolo_outputs,
    letterbox_image,
)
from services.api_client import (
    PlateApiClient,
    PlateCheckStatus,
    parse_server_plate_response,
)


def test_letterbox_preserves_aspect_and_range() -> None:
    img = Image.new("RGB", (800, 400), color=(255, 128, 0))
    tensor, scale, pad_x, pad_y = letterbox_image(img, target_size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert abs(scale - 0.8) < 1e-6
    assert pad_x == 0
    assert pad_y == 160


def test_decode_yolo26_end2end_fused_format() -> None:
    # خروجی رسمی YOLO26: [1, 300, 6] => [x1, y1, x2, y2, conf, cls]
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [100.0, 200.0, 300.0, 260.0, 0.91, float(CLASS_ID_PLATE)]
    out[0, 1] = [10.0, 10.0, 50.0, 50.0, 0.20, 0.0]  # زیر آستانه اطمینان

    dets = decode_yolo_outputs(
        outputs=[out],
        conf_threshold=0.5,
        iou_threshold=0.45,
        input_size=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        orig_w=640,
        orig_h=640,
    )
    assert len(dets) == 1
    assert dets[0].class_id == CLASS_ID_PLATE
    assert abs(dets[0].x1 - 100.0) < 1e-3
    assert abs(dets[0].y2 - 260.0) < 1e-3


def test_decode_yolo26_end2end_split_format() -> None:
    boxes = np.zeros((1, 10, 4), dtype=np.float32)
    logits = np.full((1, 10, 2), -10.0, dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.25, 0.1]  # cx, cy, w, h نرمال‌شده
    logits[0, 0, 1] = 4.0  # sigmoid(4.0) ~= 0.982

    dets = decode_yolo_outputs(
        outputs=[logits, boxes],
        conf_threshold=0.5,
        iou_threshold=0.45,
        input_size=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        orig_w=640,
        orig_h=640,
    )
    assert len(dets) == 1
    assert dets[0].class_id == 1
    assert dets[0].confidence > 0.95


def test_decode_yolov8_standard_with_nms() -> None:
    # فرمت کلاسیک YOLOv8/11 به شکل [1, 4+2, 8400]: دو کادر همپوشان از کلاس پلاک -> NMS یکی را نگه می‌دارد
    raw = np.zeros((1, 6, 8400), dtype=np.float32)
    # کادر 0 (اطمینان بالاتر)
    raw[0, :, 0] = [320.0, 320.0, 120.0, 40.0, 0.05, 0.92]
    # کادر 1 (همپوشان با کادر 0، اطمینان کمتر)
    raw[0, :, 1] = [322.0, 321.0, 120.0, 40.0, 0.05, 0.78]
    # کادر 2 (اطمینان زیر آستانه)
    raw[0, :, 2] = [100.0, 100.0, 80.0, 30.0, 0.10, 0.12]

    dets = decode_yolo_outputs(
        outputs=[raw],
        conf_threshold=0.5,
        iou_threshold=0.45,
        input_size=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        orig_w=640,
        orig_h=640,
    )
    assert len(dets) == 1
    assert dets[0].class_id == CLASS_ID_PLATE
    assert abs(dets[0].confidence - 0.92) < 1e-4


def test_decode_yolov8_transposed_format_with_nms() -> None:
    # فرمت [1, 8400, 4+2]
    raw = np.zeros((1, 8400, 6), dtype=np.float32)
    raw[0, 10] = [200.0, 300.0, 140.0, 40.0, 0.02, 0.88]
    raw[0, 11] = [201.0, 300.0, 140.0, 40.0, 0.02, 0.72]

    dets = decode_yolo_outputs(
        outputs=[raw],
        conf_threshold=0.5,
        iou_threshold=0.45,
        input_size=640,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        orig_w=640,
        orig_h=640,
    )
    assert len(dets) == 1
    assert dets[0].class_id == CLASS_ID_PLATE
    assert abs(dets[0].confidence - 0.88) < 1e-4


def test_two_stage_plate_engine_sorts_persian_chars_left_to_right() -> None:
    class MockCarsSession:
        def detect(self, _img, conf_threshold, iou_threshold=0.45):
            return [Detection(100, 200, 420, 290, 0.95, CLASS_ID_PLATE)]

    class MockCharsSession:
        def detect(self, _crop, conf_threshold, iou_threshold=0.45):
            # کاراکترها به‌صورت نامرتب برگردانده می‌شوند؛ موتور باید از چپ به راست مرتب کند:
            # 1 2 ب 3 4 5 6 7 -> کلاس‌ها در ir_lpr.yaml: 1,2,13('ب'),3,4,5,6,7
            return [
                Detection(240, 10, 270, 70, 0.91, 7),
                Detection(10, 10, 40, 70, 0.95, 1),
                Detection(80, 10, 115, 70, 0.93, 13),  # ب
                Detection(45, 10, 75, 70, 0.94, 2),
                Detection(120, 10, 150, 70, 0.92, 3),
                Detection(205, 10, 235, 70, 0.90, 6),
                Detection(155, 10, 180, 70, 0.89, 4),
                Detection(182, 10, 203, 70, 0.88, 5),
            ]

    engine = PersianPlateEngine(settings=AppSettings())
    engine.cars_model = MockCarsSession()
    engine.chars_model = MockCharsSession()
    engine.is_ready = True

    frame = Image.new("RGB", (640, 480), color=(20, 20, 20))
    res = engine.process_image(frame)

    assert len(res.plates) == 1
    plate = res.plates[0]
    assert plate.complete is True
    assert plate.plate_persian == "12ب34567"
    assert plate.plate_latin == "12B34567"
    assert plate.formatted_parts == ("۱۲", "ب", "۳۴۵", "۶۷")
    assert plate.plate_crop_jpeg is not None and len(plate.plate_crop_jpeg) > 100


def test_geometry_rotation_and_canvas_mapping() -> None:
    deg = compute_android_rotation_degrees(sensor_orientation=90, device_orientation_degrees=0)
    assert deg == 90

    landscape = Image.new("RGB", (640, 480))
    upright = rotate_pil_clockwise(landscape, 90)
    assert upright.size == (480, 640)

    x, y, w, h = normalized_box_to_canvas((0.25, 0.5, 0.75, 0.75), 400.0, 200.0)
    assert (x, y, w, h) == (100.0, 100.0, 200.0, 50.0)


def test_server_response_parsing_red_and_green_rules() -> None:
    # پاسخ متنی NO -> قرمز (REJECTED)
    r_no = parse_server_plate_response(200, "NO", "12ب34567", "12B34567", 25.0)
    assert r_no.status == PlateCheckStatus.REJECTED

    # پاسخ JSON با status=no -> قرمز (REJECTED)
    r_json_no = parse_server_plate_response(
        200, '{"status": "no", "reason": "تخلف طرح ترافیک"}', "12ب34567", "12B34567", 30.0
    )
    assert r_json_no.status == PlateCheckStatus.REJECTED

    # پاسخ غیر از NO -> سبز (APPROVED)
    r_yes = parse_server_plate_response(200, "YES", "12ب34567", "12B34567", 22.0)
    assert r_yes.status == PlateCheckStatus.APPROVED

    r_json_ok = parse_server_plate_response(
        200, '{"status": "OK", "owner": "شرکت نمونه"}', "12ب34567", "12B34567", 18.0
    )
    assert r_json_ok.status == PlateCheckStatus.APPROVED
    assert r_json_ok.details["owner"] == "شرکت نمونه"

    # خطای 500 سرور -> ERROR (کادر آبی باقی می‌ماند)
    r_err = parse_server_plate_response(500, "Internal Server Error", "12ب34567", "12B34567", 50.0)
    assert r_err.status == PlateCheckStatus.ERROR


def test_api_client_caching_avoids_duplicate_requests() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert str(request.url) == "http://myweb.web.com/api/plate/check"
        return httpx.Response(200, json={"status": "NO", "owner": "خودرو غیرمجاز"})

    async def run() -> None:
        settings = AppSettings(api_cache_ttl_seconds=30.0)
        client = PlateApiClient(settings)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            res1 = await client.check_plate("12ب34567", "12B34567", confidence=0.92)
            res2 = await client.check_plate("12ب34567", "12B34567", confidence=0.92)
            assert res1.status == PlateCheckStatus.REJECTED
            assert res2.status == PlateCheckStatus.REJECTED
            assert call_count == 1  # دومین فراخوانی از کش پاسخ داده شد
        finally:
            await client.close()

    asyncio.run(run())


def test_format_iranian_plate_helper() -> None:
    assert format_iranian_plate(["4", "5", "ص", "7", "8", "9", "1", "1"]) == ("۴۵", "ص", "۷۸۹", "۱۱")


def test_submit_violation_sends_to_api_and_logs_locally(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path))
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert str(request.url) == "http://myweb.web.com/api/violation"
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        client = PlateApiClient(AppSettings())
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            res = await client.submit_violation(
                plate_persian="۱۲ ب ۳۴۵ | ایران ۶۷",
                plate_latin="12B34567",
                violation_type="توقف ممنوع / پارک دوبل",
                description="توقف در پیاده‌رو",
                plate_crop_jpeg=b"\xff\xd8\xff\xd9",
            )
            assert res.success is True
            assert res.saved_locally is True
            assert captured_payload["plate_latin"] == "12B34567"
            assert "image_base64" in captured_payload
            log_file = tmp_path / "violations_log.jsonl"
            assert log_file.is_file()
            assert "12B34567" in log_file.read_text(encoding="utf-8")
        finally:
            await client.close()

    asyncio.run(run())

