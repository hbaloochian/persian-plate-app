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

import pytest

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
    delete_violation_record,
    load_violation_records,
    parse_server_plate_response,
    read_violation_image_bytes,
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
    # پاسخ JSON با status=no -> قرمز (REJECTED)
    r_json_no = parse_server_plate_response(
        200,
        '{"status": "no", "passenger": 3, "description": "تخلف طرح ترافیک"}',
        "12ب34567",
        "12B34567",
        30.0,
    )
    assert r_json_no.status == PlateCheckStatus.REJECTED
    assert r_json_no.details["status"] == "no"
    assert r_json_no.details["passenger"] == 3

    # پاسخ JSON با status=yes -> سبز (APPROVED)
    r_json_yes = parse_server_plate_response(
        200,
        '{"status": "yes", "passenger": 5, "description": "مجوز دارد"}',
        "12ب34567",
        "12B34567",
        22.0,
    )
    assert r_json_yes.status == PlateCheckStatus.APPROVED
    assert r_json_yes.details["status"] == "yes"

    # پاسخ متن ساده NO (نامعتبر در قرارداد جدید) -> ERROR
    r_text_no = parse_server_plate_response(200, "NO", "12ب34567", "12B34567", 25.0)
    assert r_text_no.status == PlateCheckStatus.ERROR

    # پاسخ JSON با status نامعتبر -> ERROR
    r_bad_status = parse_server_plate_response(
        200, '{"status": "OK", "owner": "شرکت نمونه"}', "12ب34567", "12B34567", 18.0
    )
    assert r_bad_status.status == PlateCheckStatus.ERROR

    # پاسخ JSON نامعتبر (غیرشیء) -> ERROR
    r_bad_json = parse_server_plate_response(200, "[1,2,3]", "12ب34567", "12B34567", 15.0)
    assert r_bad_json.status == PlateCheckStatus.ERROR

    # passenger نامعتبر: رنگ فقط از status تأثیر می‌گیرد -> سبز (APPROVED)
    r_bad_passenger = parse_server_plate_response(
        200, '{"status": "yes", "passenger": 150}', "12ب34567", "12B34567", 12.0
    )
    assert r_bad_passenger.status == PlateCheckStatus.APPROVED

    # description نامعتبر: رنگ فقط از status تأثیر می‌گیرد -> سبز (APPROVED)
    r_bad_desc = parse_server_plate_response(
        200, '{"status": "yes", "description": 123}', "12ب34567", "12B34567", 11.0
    )
    assert r_bad_desc.status == PlateCheckStatus.APPROVED

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


def test_submit_violation_with_evidence_photo_history_retry_and_delete(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path))
    captured_payloads: list[dict] = []
    server_status = 503

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        assert str(request.url) == "http://myweb.web.com/api/violation"
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        if server_status == 200:
            return httpx.Response(200, json={"ok": True, "id": "srv-1"})
        return httpx.Response(server_status, text="Service Unavailable")

    async def run() -> None:
        nonlocal server_status
        plate_jpg = b"\xff\xd8\xff\xe0PLATE_BYTES\xff\xd9"
        evidence_jpg = b"\xff\xd8\xff\xe0EVIDENCE_DOC_BYTES\xff\xd9"

        client = PlateApiClient(AppSettings())
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            # 1. ثبت تخلف با هر دو عکس (پلاک + مستندات) در حالت خطای سرور -> ذخیره محلی
            res = await client.submit_violation(
                plate_persian="۱۲ ب ۳۴۵ | ایران ۱۱",
                plate_latin="12B34511",
                violation_type="ورود غیرمجاز به محدوده",
                description="تردد در طرح ترافیک بدون مجوز",
                plate_crop_jpeg=plate_jpg,
                evidence_image_jpeg=evidence_jpg,
            )
            assert res.success is False
            assert res.saved_locally is True
            assert res.record_id
            assert captured_payloads[0]["has_image"] is True
            assert captured_payloads[0]["has_evidence_image"] is True
            assert "image_base64" in captured_payloads[0]
            assert "evidence_image_base64" in captured_payloads[0]

            # 2. بازبینی فهرست تخلفات و بررسی تصاویر ذخیره‌شده محلی
            records = load_violation_records()
            assert len(records) == 1
            rec = records[0]
            assert rec.id == res.record_id
            assert rec.synced is False
            assert rec.has_image is True
            assert rec.has_evidence_image is True
            assert read_violation_image_bytes(rec.plate_image_file) == plate_jpg
            assert read_violation_image_bytes(rec.evidence_image_file) == evidence_jpg

            # 3. ارسال مجدد پس از وصل شدن سرور -> تغییر وضعیت به synced=True
            server_status = 200
            retry_res = await client.retry_violation_sync(rec)
            assert retry_res.success is True
            refreshed = load_violation_records()
            assert len(refreshed) == 1
            assert refreshed[0].synced is True
            assert "evidence_image_base64" in captured_payloads[1]

            # 4. حذف رکورد و پاک‌سازی فایل‌های تصویر
            assert delete_violation_record(rec.id) is True
            assert load_violation_records() == []
            assert read_violation_image_bytes(rec.plate_image_file) is None
            assert read_violation_image_bytes(rec.evidence_image_file) is None
        finally:
            await client.close()

    asyncio.run(run())


def test_iranian_plate_widget_rtl_order_when_flet_available() -> None:
    ft = pytest.importorskip("flet")
    from ui.components import build_iranian_plate_widget

    widget = build_iranian_plate_widget(
        ["1", "2", "ب", "3", "4", "5", "1", "1"],
        PlateCheckStatus.APPROVED,
    )
    outer_row = widget.content
    assert isinstance(outer_row, ft.Row)
    assert outer_row.rtl is True
    assert len(outer_row.controls) == 4

    # 1. راست‌ترین بخش (controls[0]): کادر «ایران ۱۱»
    iran_col = outer_row.controls[0].content
    assert iran_col.controls[0].value == "ایران"
    assert iran_col.controls[1].value == "۱۱"

    # 2. بخش وسط (controls[2]): در RTL از راست به چپ -> ۳ رقم، حرف، ۲ رقم
    mid_row = outer_row.controls[2].content
    assert isinstance(mid_row, ft.Row)
    assert mid_row.rtl is True
    assert [c.value for c in mid_row.controls] == ["۳۴۵", "ب", "۱۲"]

    # 3. چپ‌ترین بخش (controls[3]): نوار آبی I.R. IRAN
    blue_strip = outer_row.controls[3]
    assert blue_strip.bgcolor == ft.Colors.BLUE_900


def test_jalali_calendar_conversion_and_iso_fallback() -> None:
    from datetime import datetime
    from core.jalali import (
        IRAN_TIMEZONE,
        format_jalali_date,
        format_jalali_time,
        format_jalali_verbose_date,
        gregorian_to_jalali,
        iso_to_jalali_parts,
    )

    # نوروز ۱۴۰۳ (سال کبیسه شمسی) و روز ۳۰ اسفند ۱۴۰۳ و نوروز ۱۴۰۴ و ۱۴۰۵
    assert gregorian_to_jalali(2024, 3, 20) == (1403, 1, 1)
    assert gregorian_to_jalali(2025, 3, 20) == (1403, 12, 30)
    assert gregorian_to_jalali(2025, 3, 21) == (1404, 1, 1)
    assert gregorian_to_jalali(2026, 3, 21) == (1405, 1, 1)
    assert gregorian_to_jalali(2026, 11, 14) == (1405, 8, 23)

    dt = datetime(2026, 11, 14, 15, 42, 18, tzinfo=IRAN_TIMEZONE)
    assert format_jalali_date(dt) == "1405/08/23"
    assert format_jalali_date(dt, persian_digits=True) == "۱۴۰۵/۰۸/۲۳"
    assert format_jalali_time(dt) == "15:42:18"
    assert format_jalali_verbose_date(dt) == "شنبه ۲۳ آبان ۱۴۰۵"

    # تبدیل رکوردهای قدیمی با زمان UTC به وقت ایران (UTC+03:30)
    j_date, j_time, j_verbose = iso_to_jalali_parts("2026-11-14T12:12:18+00:00")
    assert j_date == "1405/08/23"
    assert j_time == "15:42:18"
    assert j_verbose == "شنبه ۲۳ آبان ۱۴۰۵"


def test_settings_auto_query_default_disabled_and_persisted(tmp_path) -> None:
    from config import load_settings, save_settings

    # پیش‌فرض باید غیرفعال باشد
    initial = load_settings(data_dir=tmp_path)
    assert initial.auto_query_enabled is False

    # ذخیره فعال‌سازی توسط کاربر و خواندن مجدد
    initial.auto_query_enabled = True
    initial.api_base_url = "https://myweb.web.com"
    save_settings(initial, data_dir=tmp_path)

    reloaded = load_settings(data_dir=tmp_path)
    assert reloaded.auto_query_enabled is True
    assert reloaded.api_base_url == "https://myweb.web.com"


def test_iranian_plate_parse_and_compose_roundtrip() -> None:
    from config import compose_iranian_plate, parse_iranian_plate_text

    l2, letter, m3, i2 = parse_iranian_plate_text("۱۲ ب ۳۴۵ | ایران ۱۱")
    assert (l2, letter, m3, i2) == ("۱۲", "ب", "۳۴۵", "۱۱")

    display_fa, latin, chars_8 = compose_iranian_plate("12", "ب", "345", "11")
    assert display_fa == "۱۲ ب ۳۴۵ | ایران ۱۱"
    assert latin == "12B34511"
    assert chars_8 == ["1", "2", "ب", "3", "4", "5", "1", "1"]


def test_persian_text_shaper_and_bidi_visual_order() -> None:
    from core.persian_text import reshape_persian, to_visual_rtl

    # لیگچر «لا» در کلمه «پلاک»
    reshaped = reshape_persian("پلاک")
    assert "\ufefc" in reshaped or "\ufefb" in reshaped

    # حفظ جهت چپ‌به‌راست تاریخ و کد لاتین درون متن فارسی
    visual = to_visual_rtl("تاریخ: ۱۴۰۵/۰۸/۲۳ - کد 12B34511")
    assert "۱۴۰۵/۰۸/۲۳" in visual
    assert "12B34511" in visual


def test_violation_stores_jalali_datetime_and_exports_valid_pdf(tmp_path, monkeypatch) -> None:
    import io
    from PIL import Image
    from services.pdf_report import (
        build_violation_pdf_filename,
        generate_violation_pdf_bytes,
        save_pdf_backup_in_app_storage,
    )

    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path))
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"ok": True})

    # ساخت دو تصویر JPEG معتبر برای پلاک و مستندات
    plate_buf = io.BytesIO()
    Image.new("RGB", (240, 60), (240, 240, 240)).save(plate_buf, format="JPEG")
    evidence_buf = io.BytesIO()
    Image.new("RGB", (320, 240), (40, 80, 120)).save(evidence_buf, format="JPEG")

    async def run() -> None:
        client = PlateApiClient(AppSettings())
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            res = await client.submit_violation(
                plate_persian="۱۲ ب ۳۴۵ | ایران ۱۱",
                plate_latin="12B34511",
                violation_type="توقف ممنوع / پارک دوبل",
                description="پارک در محل ممنوعه",
                plate_crop_jpeg=plate_buf.getvalue(),
                evidence_image_jpeg=evidence_buf.getvalue(),
            )
            assert res.success is True
            assert captured["jalali_date"].count("/") == 2
            assert captured["jalali_time"].count(":") == 2

            records = load_violation_records()
            assert len(records) == 1
            rec = records[0]
            assert rec.jalali_date == captured["jalali_date"]
            assert rec.jalali_time == captured["jalali_time"]
            assert rec.jalali_weekday

            # نام فایل PDF باید دقیقاً بر اساس اطلاعات پلاک باشد و بدون کاراکتر غیرمجاز
            filename = build_violation_pdf_filename(rec)
            assert filename == "۱۲-ب-۳۴۵-ایران-۱۱.pdf"

            pdf_bytes = generate_violation_pdf_bytes(
                rec,
                plate_image_bytes=read_violation_image_bytes(rec.plate_image_file),
                evidence_image_bytes=read_violation_image_bytes(rec.evidence_image_file),
            )
            assert pdf_bytes.startswith(b"%PDF-")
            assert len(pdf_bytes) > 10_000

            backup_file = save_pdf_backup_in_app_storage(pdf_bytes, filename)
            assert backup_file.is_file()
            assert backup_file.name == filename
        finally:
            await client.close()

    asyncio.run(run())


def test_auth_service_single_admin_login_function() -> None:
    import inspect
    from services import auth_service

    # در ماژول احراز هویت باید فقط یک تابع (authenticate_user) وجود داشته باشد
    defined_funcs = [
        name
        for name, obj in inspect.getmembers(auth_service, inspect.isfunction)
        if obj.__module__ == auth_service.__name__
    ]
    assert defined_funcs == ["authenticate_user"]
    assert not hasattr(auth_service, "authenticate_custom")

    ok, _ = auth_service.authenticate_user("admin", "1")
    assert ok is True
    bad_ok, _ = auth_service.authenticate_user("admin", "wrong")
    assert bad_ok is False
    empty_ok, _ = auth_service.authenticate_user("", "")
    assert empty_ok is False


def test_plate_query_filter_ein_only_by_default_and_settings_persistence(tmp_path) -> None:
    from config import is_public_ein_plate, load_settings, save_settings, should_query_plate

    # پیش‌فرض: استعلام تمام پلاکها غیرفعال (فقط حرف «ع»)، اثر انگشت غیرفعال و تم شب فعال
    s = load_settings(data_dir=tmp_path)
    assert s.query_all_plates is False
    assert s.biometric_login_enabled is False
    assert s.dark_mode is True

    ein_chars = ["1", "2", "ع", "3", "4", "5", "1", "1"]
    private_chars = ["1", "2", "ب", "3", "4", "5", "1", "1"]

    assert is_public_ein_plate(chars_persian=ein_chars) is True
    assert is_public_ein_plate(chars_persian=private_chars) is False

    # در حالت پیش‌فرض فقط پلاک‌های حرف «ع» مجاز به استعلام هستند
    assert should_query_plate(s, chars_persian=ein_chars, plate_latin="12EIN34511") is True
    assert should_query_plate(s, chars_persian=private_chars, plate_latin="12B34511") is False

    # با فعال‌سازی سوییچ‌ها و تغییر به تم روز
    s.query_all_plates = True
    s.biometric_login_enabled = True
    s.dark_mode = False
    save_settings(s, data_dir=tmp_path)

    reloaded = load_settings(data_dir=tmp_path)
    assert reloaded.query_all_plates is True
    assert reloaded.biometric_login_enabled is True
    assert reloaded.dark_mode is False
    assert should_query_plate(reloaded, chars_persian=private_chars, plate_latin="12B34511") is True


def test_multiframe_plate_stabilizer_corrects_noise_and_switches_vehicles() -> None:
    from core.plate_stabilizer import MultiFramePlateStabilizer

    stab = MultiFramePlateStabilizer(window_size=5, min_confirmations=2, max_age_sec=3.0)

    # فریم اول: هنوز تأیید چندفریمی نشده است
    r1 = stab.update(["1", "2", "ع", "3", "4", "5", "1", "1"], 0.85, now=10.0)
    assert r1 is not None
    assert r1.plate_latin == "12EIN34511"
    assert r1.is_confirmed is False

    # فریم دوم منطبق: پلاک تثبیت می‌شود
    r2 = stab.update(["1", "2", "ع", "3", "4", "5", "1", "1"], 0.90, now=10.4)
    assert r2 is not None
    assert r2.is_confirmed is True
    assert r2.votes_count == 2

    # فریم سوم با خطای تک‌فریمی OCR (رقم ۳ اشتباهاً ۸ خوانده شده): رأی‌گیری اکثریت آن را اصلاح می‌کند
    r3 = stab.update(["1", "2", "ع", "8", "4", "5", "1", "1"], 0.72, now=10.8)
    assert r3 is not None
    assert r3.plate_latin == "12EIN34511"
    assert r3.is_confirmed is True
    assert r3.votes_count == 3

    # یک فریم کاملاً متفاوت (نویز یا عبور لحظه‌ای خودروی دیگر): نادیده گرفته می‌شود تا نمایشگر پرش نکند
    r_glitch = stab.update(["9", "9", "ب", "7", "6", "5", "4", "4"], 0.65, now=11.1)
    assert r_glitch is not None
    assert r_glitch.plate_latin == "12EIN34511"
    assert r_glitch.is_glitch_ignored is True

    # فریم دوم متوالی از خودروی جدید: بافر سریعاً روی خودروی جدید سوئیچ می‌کند
    r_switch = stab.update(["9", "9", "ب", "7", "6", "5", "4", "4"], 0.91, now=11.5)
    assert r_switch is not None
    assert r_switch.plate_latin == "99B76544"
    assert r_switch.is_confirmed is True
    assert r_switch.is_glitch_ignored is False

    # پس از گذشت زمان max_age_sec فریم‌های قدیمی منقضی می‌شوند
    r_fresh = stab.update(["5", "5", "ص", "1", "1", "1", "2", "2"], 0.88, now=20.0)
    assert r_fresh is not None
    assert r_fresh.plate_latin == "55SAD11122"
    assert r_fresh.votes_count == 1
    assert r_fresh.is_confirmed is False


def test_violation_gps_storage_retry_and_pdf_report(tmp_path, monkeypatch) -> None:
    import json
    from services.pdf_report import generate_violation_pdf_bytes

    monkeypatch.setenv("FLET_APP_STORAGE_DATA", str(tmp_path))
    captured_payloads: list[dict] = []
    status_code = 503

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(status_code, text="OK" if status_code == 200 else "ERR")

    async def run() -> None:
        nonlocal status_code
        client = PlateApiClient(AppSettings())
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            res = await client.submit_violation(
                plate_persian="۱۲ ع ۳۴۵ | ایران ۱۱",
                plate_latin="12EIN34511",
                violation_type="حرکت خارج از مسیر راهنامه/پروانه",
                description="خروج از مسیر مجاز جاده‌ای",
                latitude=35.689214,
                longitude=51.389012,
                location_accuracy_m=11.8,
            )
            assert res.success is False
            assert captured_payloads[0]["latitude"] == pytest.approx(35.689214)
            assert captured_payloads[0]["longitude"] == pytest.approx(51.389012)
            assert captured_payloads[0]["location_accuracy_m"] == pytest.approx(11.8)

            records = load_violation_records()
            assert len(records) == 1
            rec = records[0]
            assert rec.has_gps is True
            assert "35.689214, 51.389012" in rec.formatted_gps
            assert "±12m" in rec.formatted_gps
            assert rec.google_maps_url == "https://maps.google.com/?q=35.689214,51.389012"

            # ارسال مجدد باید مختصات GPS را حفظ کند
            status_code = 200
            retry_res = await client.retry_violation_sync(rec)
            assert retry_res.success is True
            assert captured_payloads[1]["latitude"] == pytest.approx(35.689214)

            # گزارش PDF تک‌تخلف با حضور GPS باید معتبر تولید شود
            pdf_bytes = generate_violation_pdf_bytes(rec)
            assert pdf_bytes.startswith(b"%PDF-")
            assert len(pdf_bytes) > 10_000
        finally:
            await client.close()

    asyncio.run(run())


def test_violation_history_search_filter_and_batch_pdf_csv_exports() -> None:
    from services.api_client import ViolationRecord, filter_violation_records
    from services.pdf_report import (
        build_batch_export_filename,
        generate_batch_violations_csv_bytes,
        generate_batch_violations_pdf_bytes,
        generate_batch_violations_pdf_images,
    )

    records = [
        ViolationRecord(
            id=f"v-{i}",
            plate=f"{10 + i} ع {300 + i} | ایران ۱۱" if i % 2 == 0 else f"{20 + i} ب {400 + i} | ایران ۲۲",
            plate_latin=f"{10 + i}EIN{300 + i}11" if i % 2 == 0 else f"{20 + i}B{400 + i}22",
            violation_type="اضافه تناژ" if i % 3 == 0 else "نداشتن صورت وضعیت مسافری",
            description=f"یادداشت شماره {i}",
            timestamp="2026-11-14T10:00:00+00:00",
            jalali_date="1405/08/23",
            jalali_time=f"13:{i:02d}:00",
            latitude=35.7000 + i * 0.001 if i % 2 == 0 else None,
            longitude=51.4000 + i * 0.001 if i % 2 == 0 else None,
            location_accuracy_m=10.0 if i % 2 == 0 else None,
            synced=(i % 2 == 0),
        )
        for i in range(1, 23)
    ]

    # فیلتر وضعیت ارسال و GPS
    assert len(filter_violation_records(records, sync_filter="synced")) == 11
    assert len(filter_violation_records(records, sync_filter="pending")) == 11
    assert len(filter_violation_records(records, sync_filter="gps")) == 11

    # فیلتر نوع تخلف
    overweight = filter_violation_records(records, violation_type="اضافه تناژ")
    assert len(overweight) == 7
    assert all(r.violation_type == "اضافه تناژ" for r in overweight)

    # جستجو با ارقام فارسی منطبق با کد لاتین/پلاک
    found = filter_violation_records(records, search_query="۱۲ ع ۳۰۲")
    assert len(found) == 1
    assert found[0].id == "v-2"

    # نام فایل خروجی گروهی
    assert build_batch_export_filename("pdf").endswith(".pdf")
    assert build_batch_export_filename("csv").endswith(".csv")

    # خروجی PDF گروهی چندصفحه‌ای (۲۲ رکورد -> حداقل ۲ صفحه A4)
    pages = generate_batch_violations_pdf_images(records, filter_summary="فیلتر: همه تخلفات")
    assert len(pages) >= 2
    batch_pdf = generate_batch_violations_pdf_bytes(records, filter_summary="فیلتر: همه تخلفات")
    assert batch_pdf.startswith(b"%PDF-")
    assert len(batch_pdf) > 20_000

    # خروجی اکسل CSV با UTF-8 BOM
    csv_bytes = generate_batch_violations_csv_bytes(records)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    csv_text = csv_bytes.decode("utf-8-sig")
    assert "شماره پلاک فارسی" in csv_text
    assert "عرض جغرافیایی (Lat)" in csv_text
    assert "12EIN30211" in csv_text
    assert "https://maps.google.com/?q=" in csv_text





