"""موتور استنتاج محلی پلاک‌خوان ایرانی با ONNX Runtime (بدون نیاز به PyTorch و OpenCV).

ویژگی‌های مهندسی:
1. سازگاری کامل با معماری دومرحله‌ای مخزن shahabbai/Persian_Plate_Recognition:
   - مرحله 1: تشخیص خودرو و پلاک (cars_best_v26.onnx -> کلاس 0=Car، کلاس 1=Plate)
   - مرحله 2: تشخیص 39 کلاس کاراکتر فارسی روی برش پلاک (chars_best_v26.onnx)
2. پشتیبانی خودکار از هر سه فرمت خروجی ONNX در Ultralytics:
   - YOLO26 End-to-End Fused: [1, 300, 6] (بدون نیاز به NMS، فوق‌العاده سریع)
   - YOLO26 End-to-End Split: logits [1, 300, nc] + boxes [1, 300, 4]
   - YOLOv8 / YOLOv11 استاندارد: [1, 4+nc, N] یا [1, N, 4+nc] با NMS برداری سریع NumPy
3. پشتیبانی از حالت تک‌مدلی: اگر کاربر فقط یک فایل .onnx در پوشه weights قرار دهد،
   برنامه بدون خطا پلاک را تشخیص داده و کادر آن را نمایش می‌دهد.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import (
    CHAR_NAMES_LATIN,
    CHAR_NAMES_PERSIAN,
    CLASS_ID_PLATE,
    DEFAULT_CARS_MODEL_NAME,
    DEFAULT_CHARS_MODEL_NAME,
    AppSettings,
    find_weights_dir,
    format_iranian_plate,
    resolve_ort_threads,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    """یک شیء یا کاراکتر تشخیص‌داده‌شده در مختصات تصویر ورودی (پیکسل)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(slots=True)
class PlateDetection:
    """نتیجه کامل تشخیص یک پلاک در فریم زنده."""

    # مختصات نرمال‌شده در بازه [0, 1] برای رسم دقیق روی Canvas
    norm_box: tuple[float, float, float, float]
    # مختصات پیکسلی در تصویر عمودی‌شده
    pixel_box: tuple[int, int, int, int]
    confidence: float
    plate_persian: str
    plate_latin: str
    chars_persian: list[str] = field(default_factory=list)
    complete: bool = False
    plate_crop_jpeg: bytes | None = None

    @property
    def formatted_parts(self) -> tuple[str, str, str, str]:
        return format_iranian_plate(self.chars_persian)

    @property
    def display_text(self) -> str:
        if len(self.chars_persian) == 8:
            left2, letter, mid3, iran2 = self.formatted_parts
            return f"{left2} {letter} {mid3} | ایران {iran2}"
        return self.plate_persian or "پلاک شناسایی شد"


@dataclass(slots=True)
class FrameInferenceResult:
    """نتیجه استنتاج روی یک فریم کامل دوربین."""

    plates: list[PlateDetection]
    inference_ms: float
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# پیش‌پردازش و پس‌پردازش برداری فوق‌سریع با NumPy و Pillow
# ---------------------------------------------------------------------------


def letterbox_image(
    image: Image.Image,
    target_size: int = 640,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    """تصویر RGB را با حفظ نسبت ابعاد (Letterbox) به تنسور NCHW float32 تبدیل می‌کند.

    Returns:
        tensor: آرایه [1, 3, target_size, target_size] با مقادیر [0.0, 1.0]
        scale: ضریب مقیاس استفاده‌شده
        pad_x: حاشیه چپ (پیکسل)
        pad_y: حاشیه بالا (پیکسل)
    """
    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) * (1.0 / 255.0)
    # HWC -> NCHW پیوسته در حافظه برای حداکثر سرعت ONNX Runtime
    tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, :, :, :])
    return tensor, scale, pad_x, pad_y


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """الگوریتم Non-Maximum Suppression برداری سریع با NumPy."""
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_threshold]

    return keep


def _scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """تبدیل مختصات کادر از فضای Letterbox به فضای تصویر اصلی."""
    scaled = boxes_xyxy.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - pad_x) / scale
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - pad_y) / scale
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, float(orig_w))
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, float(orig_h))
    return scaled


def decode_yolo_outputs(
    outputs: list[np.ndarray],
    conf_threshold: float,
    iou_threshold: float,
    input_size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> list[Detection]:
    """خروجی خام ONNX را به‌صورت خودکار برای YOLO26 (End2End) و YOLOv8/11 دیکد می‌کند."""
    if not outputs:
        return []

    # حالت 1: خروجی دوگانه YOLO26 End2End Split (logits + normalized cxcywh boxes)
    if len(outputs) >= 2 and outputs[0].ndim == 3 and outputs[1].ndim == 3:
        a, b = outputs[0][0], outputs[1][0]
        if a.shape[0] == b.shape[0] and (a.shape[1] == 4 or b.shape[1] == 4):
            boxes_norm, logits = (a, b) if a.shape[1] == 4 else (b, a)
            # اگر مقادیر logits خارج از [0, 1] باشند، سیگموید اعمال می‌کنیم
            probs = _sigmoid(logits) if (logits.min() < 0.0 or logits.max() > 1.0) else logits
            class_ids = np.argmax(probs, axis=1)
            scores = probs[np.arange(probs.shape[0]), class_ids]
            mask = scores >= conf_threshold
            if not np.any(mask):
                return []

            boxes_norm = boxes_norm[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            # cx, cy, w, h نرمال‌شده [0..1] -> xyxy در فضای input_size
            cx = boxes_norm[:, 0] * input_size
            cy = boxes_norm[:, 1] * input_size
            bw = boxes_norm[:, 2] * input_size
            bh = boxes_norm[:, 3] * input_size
            xyxy = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)
            xyxy = _scale_boxes_to_original(xyxy, scale, pad_x, pad_y, orig_w, orig_h)
            return [
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(conf),
                    class_id=int(cls_id),
                )
                for box, conf, cls_id in zip(xyxy, scores, class_ids)
                if (box[2] - box[0]) >= 2.0 and (box[3] - box[1]) >= 2.0
            ]

    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2 or raw.size == 0:
        return []

    # حالت 2: YOLO26 End-to-End Fused -> شکل [300, 6] => [x1, y1, x2, y2, conf, cls]
    # (خروجی رسمی export_onnx.py در مخزن shahabbai/Persian_Plate_Recognition)
    if raw.shape[1] == 6 and raw.shape[0] <= 1000:
        scores = raw[:, 4]
        mask = scores >= conf_threshold
        if not np.any(mask):
            return []
        filtered = raw[mask]
        xyxy = filtered[:, :4].astype(np.float32)
        # اگر مدل خروجی نرمال‌شده [0..1] داده باشد، به فضای input_size مقیاس می‌دهیم
        if xyxy.max() <= 1.5:
            xyxy = xyxy * float(input_size)
        xyxy = _scale_boxes_to_original(xyxy, scale, pad_x, pad_y, orig_w, orig_h)
        dets = [
            Detection(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(row[4]),
                class_id=int(round(float(row[5]))),
            )
            for box, row in zip(xyxy, filtered)
            if (box[2] - box[0]) >= 2.0 and (box[3] - box[1]) >= 2.0
        ]
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets

    # حالت 3: YOLOv8 / YOLOv11 استاندارد -> شکل [4 + nc, N] یا [N, 4 + nc]
    pred = raw.T if raw.shape[0] < raw.shape[1] else raw
    if pred.shape[1] < 5:
        return []

    box_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_threshold
    if not np.any(mask):
        return []

    box_cxcywh = box_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    cx, cy, bw, bh = box_cxcywh[:, 0], box_cxcywh[:, 1], box_cxcywh[:, 2], box_cxcywh[:, 3]
    xyxy = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)
    if xyxy.max() <= 1.5:
        xyxy = xyxy * float(input_size)
    xyxy = _scale_boxes_to_original(xyxy, scale, pad_x, pad_y, orig_w, orig_h)

    # اعمال NMS به تفکیک هر کلاس
    results: list[Detection] = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        cls_boxes = xyxy[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms_boxes(cls_boxes, cls_scores, iou_threshold=iou_threshold)
        for idx in keep:
            b = cls_boxes[idx]
            if (b[2] - b[0]) >= 2.0 and (b[3] - b[1]) >= 2.0:
                results.append(
                    Detection(
                        x1=float(b[0]),
                        y1=float(b[1]),
                        x2=float(b[2]),
                        y2=float(b[3]),
                        confidence=float(cls_scores[idx]),
                        class_id=int(cls_id),
                    )
                )

    results.sort(key=lambda d: d.confidence, reverse=True)
    return results


# ---------------------------------------------------------------------------
# کلاس‌های مدیریت نشست ONNX Runtime و پایپ‌لاین دو مرحله‌ای
# ---------------------------------------------------------------------------


class OnnxYoloSession:
    """پوشش سبک و بهینه‌شده برای اجرای یک مدل YOLO روی ONNX Runtime موبایل."""

    def __init__(self, model_path: Path, input_size: int = 640, num_threads: int = 4) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = max(1, num_threads)
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # انتخاب بهترین شتاب‌دهنده سخت‌افزاری موجود روی گوشی اندروید
        available = set(ort.get_available_providers())
        preferred = ["XNNPACKExecutionProvider", "NnapiExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        self.output_names: list[str] = [o.name for o in self.session.get_outputs()]

        # تشخیص خودکار ابعاد ورودی ثابت مدل ONNX (در صورت متفاوت بودن از 640)
        if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and inp.shape[2] > 0:
            self.input_size = int(inp.shape[2])
        else:
            self.input_size = input_size

        logger.info(
            "ONNX model loaded: %s | input=%dx%d | providers=%s",
            model_path.name,
            self.input_size,
            self.input_size,
            self.session.get_providers(),
        )

    def detect(
        self,
        image: Image.Image,
        conf_threshold: float,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        orig_w, orig_h = image.size
        tensor, scale, pad_x, pad_y = letterbox_image(image, target_size=self.input_size)
        raw_outputs = self.session.run(self.output_names, {self.input_name: tensor})
        np_outputs = [np.asarray(o, dtype=np.float32) for o in raw_outputs]
        return decode_yolo_outputs(
            outputs=np_outputs,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_size=self.input_size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=orig_w,
            orig_h=orig_h,
        )


class PersianPlateEngine:
    """پایپ‌لاین کامل تشخیص پلاک و خوانش حروف فارسی (کاملاً آفلاین روی گوشی)."""

    def __init__(self, settings: AppSettings, weights_dir: Path | None = None) -> None:
        self.settings = settings
        self.weights_dir = weights_dir or find_weights_dir()
        self.cars_model: Any | None = None
        self.chars_model: Any | None = None
        self.load_error: str | None = None
        self.is_ready: bool = False
        self.single_stage_mode: bool = False

    def resolve_model_paths(self) -> tuple[Path | None, Path | None]:
        """فایل‌های ONNX درون پوشه weights را به‌صورت هوشمند شناسایی می‌کند."""
        if not self.weights_dir.is_dir():
            return None, None

        cars_default = self.weights_dir / DEFAULT_CARS_MODEL_NAME
        chars_default = self.weights_dir / DEFAULT_CHARS_MODEL_NAME
        if cars_default.is_file() and chars_default.is_file():
            return cars_default, chars_default

        onnx_files = sorted(self.weights_dir.glob("*.onnx"))
        if not onnx_files:
            return None, None

        cars_path = next(
            (p for p in onnx_files if any(k in p.stem.lower() for k in ("car", "plate", "det"))),
            None,
        )
        chars_path = next(
            (p for p in onnx_files if any(k in p.stem.lower() for k in ("char", "ocr", "lpr"))),
            None,
        )

        # اگر فقط یک فایل .onnx قرار داده شده باشد (حالت تک‌مدلی)
        if cars_path is None:
            cars_path = onnx_files[0]
        if chars_path == cars_path:
            chars_path = None

        return cars_path, chars_path

    def load(self) -> bool:
        """مدل‌های ONNX را بارگذاری می‌کند."""
        cars_path, chars_path = self.resolve_model_paths()
        if cars_path is None:
            self.load_error = (
                f"فایل مدل ONNX در پوشه '{self.weights_dir}' پیدا نشد.\n"
                f"لطفاً فایل‌های {DEFAULT_CARS_MODEL_NAME} و {DEFAULT_CHARS_MODEL_NAME} را در پوشه weights قرار دهید."
            )
            self.is_ready = False
            return False

        try:
            # تعداد تراشه‌های ORT بر اساس منابع دستگاه (و حالت مدیریت باتری) تنظیم می‌شود
            ort_threads = resolve_ort_threads(self.settings)
            self.cars_model = OnnxYoloSession(
                model_path=cars_path,
                input_size=self.settings.input_size,
                num_threads=ort_threads,
            )
            if chars_path is not None:
                self.chars_model = OnnxYoloSession(
                    model_path=chars_path,
                    input_size=self.settings.input_size,
                    num_threads=ort_threads,
                )
                self.single_stage_mode = False
            else:
                self.chars_model = None
                self.single_stage_mode = True

            self.is_ready = True
            self.load_error = None
            return True
        except Exception as exc:
            logger.exception("Failed to initialize ONNX models")
            self.load_error = f"خطا در بارگذاری مدل ONNX: {exc}"
            self.is_ready = False
            return False

    def process_image(self, image: Image.Image) -> FrameInferenceResult:
        """روی یک تصویر RGB عمودی‌شده، پلاک‌ها و کاراکترهای آن‌ها را استخراج می‌کند."""
        t0 = time.perf_counter()
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_w, img_h = image.size

        if not self.is_ready or self.cars_model is None:
            return FrameInferenceResult(
                plates=[],
                inference_ms=(time.perf_counter() - t0) * 1000.0,
                image_width=img_w,
                image_height=img_h,
            )

        stage1_dets: list[Detection] = self.cars_model.detect(
            image,
            conf_threshold=self.settings.cars_conf_threshold,
            iou_threshold=self.settings.nms_iou_threshold,
        )

        # انتخاب کادر پلاک‌ها: در مدل cars_best_v26 کلاس 1 مربوط به پلاک است.
        # اگر مدل تک‌کلاسه سفارشی باشد (کلاس 0 = پلاک)، fallback هوشمند انجام می‌شود.
        plate_dets = [d for d in stage1_dets if d.class_id == CLASS_ID_PLATE]
        if not plate_dets and self.single_stage_mode:
            plate_dets = stage1_dets

        plates: list[PlateDetection] = []
        for det in plate_dets:
            x1 = max(0, min(img_w - 1, int(det.x1)))
            y1 = max(0, min(img_h - 1, int(det.y1)))
            x2 = max(x1 + 1, min(img_w, int(det.x2)))
            y2 = max(y1 + 1, min(img_h, int(det.y2)))

            if (x2 - x1) < 12 or (y2 - y1) < 6:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crop_jpeg = self._encode_crop_jpeg(crop)

            chars_fa: list[str] = []
            chars_en: list[str] = []

            if self.chars_model is not None:
                char_dets: list[Detection] = self.chars_model.detect(
                    crop,
                    conf_threshold=self.settings.chars_conf_threshold,
                    iou_threshold=self.settings.nms_iou_threshold,
                )
                # حذف کاراکترهای همپوشان با حفظ بیشترین اطمینان و مرتب‌سازی از چپ به راست
                char_dets = self._deduplicate_chars(char_dets)
                char_dets.sort(key=lambda c: c.x1)

                for ch in char_dets:
                    if 0 <= ch.class_id < len(CHAR_NAMES_PERSIAN):
                        chars_fa.append(CHAR_NAMES_PERSIAN[ch.class_id])
                        chars_en.append(CHAR_NAMES_LATIN[ch.class_id])

            plate_persian = "".join(chars_fa)
            plate_latin = "".join(chars_en)
            complete = len(chars_fa) == self.settings.expected_plate_chars

            plates.append(
                PlateDetection(
                    norm_box=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                    pixel_box=(x1, y1, x2, y2),
                    confidence=det.confidence,
                    plate_persian=plate_persian,
                    plate_latin=plate_latin,
                    chars_persian=chars_fa,
                    complete=complete,
                    plate_crop_jpeg=crop_jpeg,
                )
            )

        # مرتب‌سازی پلاک‌ها بر اساس بیشترین اطمینان
        plates.sort(key=lambda p: (p.complete, p.confidence), reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return FrameInferenceResult(
            plates=plates,
            inference_ms=elapsed_ms,
            image_width=img_w,
            image_height=img_h,
        )

    @staticmethod
    def _deduplicate_chars(char_dets: list[Detection], iou_thresh: float = 0.45) -> list[Detection]:
        """در صورت تشخیص دو کاراکتر روی یک موقعیت مکانی، کاراکتر با اطمینان بالاتر را نگه می‌دارد."""
        if len(char_dets) <= 1:
            return char_dets
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in char_dets], dtype=np.float32)
        scores = np.array([d.confidence for d in char_dets], dtype=np.float32)
        keep = nms_boxes(boxes, scores, iou_threshold=iou_thresh)
        return [char_dets[i] for i in keep]

    @staticmethod
    def _encode_crop_jpeg(crop: Image.Image) -> bytes:
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=82)
        return buf.getvalue()
