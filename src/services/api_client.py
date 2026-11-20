"""کلاینت غیرهمزمان ارتباط با API سرور myweb.web.com با کش هوشمند، ثبت تخلف و مدیریت تاریخچه محلی."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from config import (
    AppSettings,
    get_writable_data_dir,
    to_latin_digits,
    to_persian_digits,
)
from core.jalali import (
    format_jalali_date,
    format_jalali_time,
    format_jalali_verbose_date,
    iso_to_jalali_parts,
    now_iran_datetime,
)

logger = logging.getLogger(__name__)

# حداکثر حجم مجاز پاسخ سرور (بایت). جلوگیری از پاسخ بسیار بزرگ که می‌تواند حافظه را بپوشاند.
_MAX_RESPONSE_BYTES = 1_000_000  # یک مگابایت


def _safe_error_text(exc: Exception) -> str:
    """متن خطای خنثی‌سازی‌شده برای لاگ و ذخیره‌سازی محلی.

    فقط نام کلاس خطا را نشان می‌دهد و از افشای متن کامل خطای حساسی
    (مثل URLها یا محتوای پاسخ سرور) در لاگ و فایل محلی جلوگیری می‌کند.
    """
    return f"خطای شبکه ({type(exc).__name__})"


class PlateCheckStatus(str, Enum):
    """وضعیت استعلام پلاک برای تعیین رنگ کادر دور پلاک."""

    PENDING = "pending"    # آبی: تازه شناسایی شده / در حال استعلام
    REJECTED = "rejected"  # قرمز: پاسخ سرور NO
    APPROVED = "approved"  # سبز: پاسخ سرور غیر از NO
    ERROR = "error"        # آبی + هشدار خطای شبکه در بخش پایین


@dataclass(slots=True)
class ApiCheckResult:
    """نتیجه ساختاریافته استعلام پلاک از سرور."""

    plate_persian: str
    plate_latin: str
    status: PlateCheckStatus
    raw_response: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    checked_at: str = ""
    error_message: str | None = None

    @property
    def is_no(self) -> bool:
        return self.status == PlateCheckStatus.REJECTED


@dataclass(slots=True)
class ViolationSubmissionResult:
    """نتیجه ثبت تخلف در سرور و لاگ محلی."""

    success: bool
    message: str
    saved_locally: bool = False
    server_response: str = ""
    record_id: str = ""


@dataclass(slots=True)
class ViolationRecord:
    """ساختار هر رکورد تخلف ثبت‌شده برای نمایش در صفحه بازبینی تخلفات، گزارش PDF و ارسال مجدد."""

    id: str
    plate: str
    plate_latin: str
    violation_type: str
    description: str
    timestamp: str
    local_time: str = ""
    jalali_date: str = ""
    jalali_time: str = ""
    jalali_weekday: str = ""
    latitude: float | None = None
    longitude: float | None = None
    location_accuracy_m: float | None = None
    has_image: bool = False
    has_evidence_image: bool = False
    plate_image_file: str | None = None
    evidence_image_file: str | None = None
    synced: bool = False
    server_response: str = ""

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    @property
    def formatted_gps(self) -> str:
        """نمایش خوانای مختصات جغرافیایی (مثال: 35.689200, 51.389000 (±12m))."""
        if not self.has_gps:
            return "ثبت نشده"
        base = f"{self.latitude:.6f}, {self.longitude:.6f}"
        if self.location_accuracy_m is not None and self.location_accuracy_m > 0:
            return f"{base} (±{int(round(self.location_accuracy_m))}m)"
        return base

    @property
    def google_maps_url(self) -> str:
        """لینک مستقیم نمایش موقعیت تخلف روی نقشه."""
        if not self.has_gps:
            return ""
        return f"https://maps.google.com/?q={self.latitude:.6f},{self.longitude:.6f}"

    @property
    def display_jalali_datetime(self) -> str:
        """تاریخ شمسی و ساعت را با ارقام فارسی برمی‌گرداند (مثال: ۱۴۰۵/۰۸/۲۳ - ۱۵:۴۲:۱۸)."""
        if self.jalali_date and self.jalali_time:
            return to_persian_digits(f"{self.jalali_date} - {self.jalali_time}")
        if self.jalali_date:
            return to_persian_digits(self.jalali_date)
        return to_persian_digits(self.local_time or self.timestamp)

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_index: int = 0) -> "ViolationRecord":
        raw_ts = str(data.get("timestamp") or "")
        local_time = str(data.get("local_time") or "")
        jalali_date = str(data.get("jalali_date") or "")
        jalali_time = str(data.get("jalali_time") or "")
        jalali_weekday = str(data.get("jalali_weekday") or "")

        # سازگاری با رکوردهای قدیمی که فقط timestamp میلادی داشتند
        if (not jalali_date or not jalali_time) and raw_ts:
            conv_date, conv_time, conv_verbose = iso_to_jalali_parts(raw_ts)
            jalali_date = jalali_date or conv_date
            jalali_time = jalali_time or conv_time
            jalali_weekday = jalali_weekday or conv_verbose

        if not local_time:
            if jalali_date and jalali_time:
                local_time = f"{jalali_date} - {jalali_time}"
            elif raw_ts:
                local_time = raw_ts.replace("T", " ")[:19]

        def _parse_float(val: Any) -> float | None:
            if val is None or val == "":
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        return cls(
            id=str(data.get("id") or f"legacy-{fallback_index}"),
            plate=str(data.get("plate") or "---"),
            plate_latin=str(data.get("plate_latin") or ""),
            violation_type=str(data.get("violation_type") or "تخلف ثبت‌شده"),
            description=str(data.get("description") or ""),
            timestamp=raw_ts,
            local_time=local_time,
            jalali_date=jalali_date,
            jalali_time=jalali_time,
            jalali_weekday=jalali_weekday,
            latitude=_parse_float(data.get("latitude")),
            longitude=_parse_float(data.get("longitude")),
            location_accuracy_m=_parse_float(data.get("location_accuracy_m")),
            has_image=bool(data.get("has_image", False)),
            has_evidence_image=bool(data.get("has_evidence_image", False)),
            plate_image_file=(
                str(data["plate_image_file"]) if data.get("plate_image_file") else None
            ),
            evidence_image_file=(
                str(data["evidence_image_file"]) if data.get("evidence_image_file") else None
            ),
            synced=bool(data.get("synced", False)),
            server_response=str(data.get("server_response") or ""),
        )


def get_violation_log_file() -> Path:
    """مسیر فایل JSONL تاریخچه تخلفات."""
    return get_writable_data_dir() / "violations_log.jsonl"


def get_violation_images_dir() -> Path:
    """پوشه نگهداری تصاویر برش پلاک و مستندات تخلفات ثبت‌شده."""
    path = get_writable_data_dir() / "violation_images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_violation_image_bytes(filename: str, image_bytes: bytes | None) -> str | None:
    """ذخیره امن تصویر JPEG در پوشه تصاویر تخلفات."""
    if not image_bytes:
        return None
    safe_name = Path(filename).name
    try:
        dest = get_violation_images_dir() / safe_name
        dest.write_bytes(image_bytes)
        return safe_name
    except OSError as exc:
        logger.warning("Could not save violation image '%s': %s", safe_name, exc)
        return None


def read_violation_image_bytes(filename: str | None) -> bytes | None:
    """خواندن بایت‌های تصویر ذخیره‌شده یک تخلف از حافظه محلی."""
    if not filename:
        return None
    safe_name = Path(filename).name
    path = get_violation_images_dir() / safe_name
    if not path.is_file():
        return None
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read violation image '%s': %s", safe_name, exc)
        return None


def load_violation_records() -> list[ViolationRecord]:
    """فهرست تخلفات ثبت‌شده را از جدیدترین به قدیمی‌ترین بارگذاری می‌کند."""
    log_file = get_violation_log_file()
    if not log_file.is_file():
        return []

    records: list[ViolationRecord] = []
    try:
        lines = log_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read violations log: %s", exc)
        return []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                records.append(ViolationRecord.from_dict(parsed, fallback_index=idx))
        except json.JSONDecodeError:
            logger.warning("Skipping malformed violation log line %d", idx + 1)

    records.reverse()
    return records


def _rewrite_violation_records(records_oldest_first: list[ViolationRecord]) -> bool:
    log_file = get_violation_log_file()
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as fh:
            for rec in records_oldest_first:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        logger.warning("Could not rewrite violations log: %s", exc)
        return False


def delete_violation_record(record_id: str) -> bool:
    """یک رکورد تخلف و تصاویر مرتبط با آن را از حافظه محلی حذف می‌کند."""
    records_newest = load_violation_records()
    target: ViolationRecord | None = None
    remaining: list[ViolationRecord] = []
    for rec in records_newest:
        if rec.id == record_id and target is None:
            target = rec
        else:
            remaining.append(rec)

    if target is None:
        return False

    # بازگرداندن ترتیب قدیمی به جدید برای ذخیره در JSONL
    remaining.reverse()
    ok = _rewrite_violation_records(remaining)
    if ok:
        img_dir = get_violation_images_dir()
        for fname in (target.plate_image_file, target.evidence_image_file):
            if fname:
                try:
                    (img_dir / Path(fname).name).unlink(missing_ok=True)
                except OSError:
                    pass
    return ok


def update_violation_sync_status(
    record_id: str,
    synced: bool,
    server_response: str = "",
) -> bool:
    """وضعیت همگام‌سازی یک رکورد تخلف را در فایل محلی به‌روزرسانی می‌کند."""
    records_newest = load_violation_records()
    updated = False
    for rec in records_newest:
        if rec.id == record_id:
            rec.synced = synced
            rec.server_response = server_response
            updated = True
            break
    if not updated:
        return False
    records_newest.reverse()
    return _rewrite_violation_records(records_newest)


def parse_server_plate_response(
    status_code: int,
    response_text: str,
    plate_persian: str,
    plate_latin: str,
    latency_ms: float,
) -> ApiCheckResult:
    """پاسخ متنی یا JSON سرور myweb.web.com را تحلیل و وضعیت رنگ کادر را مشخص می‌کند.

    قاعده اصلی مطابق نیازمندی:
    - در صورت پاسخ NO (به‌صورت متن ساده "NO" یا درون فیلدهای JSON مانند status/result/answer/response
      یا مقدار allowed=false) => وضعیت REJECTED (کادر قرمز)
    - در غیر این صورت (پاسخ موفق 2xx غیر از NO) => وضعیت APPROVED (کادر سبز)
    """
    now_str = datetime.now().strftime("%H:%M:%S")
    cleaned = response_text.strip()

    if status_code < 200 or status_code >= 300:
        return ApiCheckResult(
            plate_persian=plate_persian,
            plate_latin=plate_latin,
            status=PlateCheckStatus.ERROR,
            raw_response=cleaned,
            latency_ms=latency_ms,
            checked_at=now_str,
            error_message=f"خطای سرور (کد {status_code})",
        )

    details: dict[str, Any] = {}
    verdict_is_no = False

    if cleaned.upper() == "NO":
        verdict_is_no = True
        details = {"پاسخ سرور": "NO"}
    else:
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                details = parsed
                # بررسی فیلدهای رایج پاسخ در APIهای REST
                for key in ("status", "result", "response", "answer", "decision", "code"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip().upper() == "NO":
                        verdict_is_no = True
                        break
                if parsed.get("allowed") is False or parsed.get("valid") is False:
                    verdict_is_no = True
            elif isinstance(parsed, str):
                verdict_is_no = parsed.strip().upper() == "NO"
                details = {"پاسخ سرور": parsed}
            elif isinstance(parsed, bool):
                verdict_is_no = not parsed
                details = {"پاسخ سرور": "YES" if parsed else "NO"}
            else:
                details = {"پاسخ سرور": str(parsed)}
        except json.JSONDecodeError:
            verdict_is_no = cleaned.upper() == "NO"
            details = {"پاسخ سرور": cleaned or "OK"}

    return ApiCheckResult(
        plate_persian=plate_persian,
        plate_latin=plate_latin,
        status=PlateCheckStatus.REJECTED if verdict_is_no else PlateCheckStatus.APPROVED,
        raw_response=cleaned,
        details=details,
        latency_ms=latency_ms,
        checked_at=now_str,
    )


class PlateApiClient:
    """کلاینت HTTP سریع با Connection Pooling، کش TTL و جلوگیری از درخواست‌های تکراری همزمان."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, ApiCheckResult]] = {}
        self._inflight: dict[str, asyncio.Task[ApiCheckResult]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.api_timeout_seconds),
                # لرزش‌ناپذیری: درخواست POST حساسی هرگز به سرور دیگری کشه‌ندهی نشود
                follow_redirects=False,
                # اعتبارسنجی صریح TLS (رمزنگاری سرور راستی‌سنجی شود، نه نادیده گرفته شود)
                verify=True,
                headers={
                    "Accept": "application/json, text/plain;q=0.9",
                    "User-Agent": "PersianPlateScanner/1.0-Android",
                },
            )
        return self._client

    def _require_https_base(self) -> bool:
        """بررسی اجباری پروتکل HTTPS در آدرس پایه سرور.

        در صورت عدم استفاده از `https://`، با هشدار درخواست را رد می‌کند تا
        داده‌های حساسی (پلاک، GPS، تصاویر، پاسخ) هرگز به‌صورت متن‌کلا انتقال داده نشوند.
        """
        if not str(self.settings.api_base_url).lower().startswith("https://"):
            logger.warning("Blocked API request: base URL does not use HTTPS.")
            return False
        return True

    def _response_too_large(self, resp: httpx.Response) -> bool:
        """بررسی حداکثر حجم پاسخ سرور."""
        return len(resp.content) > _MAX_RESPONSE_BYTES

    def get_cached(self, plate_key: str) -> ApiCheckResult | None:
        """در صورت معتبر بودن کش برای این پلاک، نتیجه فوری برمی‌گرداند."""
        entry = self._cache.get(plate_key)
        if entry is None:
            return None
        expires_at, result = entry
        if time.monotonic() <= expires_at:
            return result
        self._cache.pop(plate_key, None)
        return None

    def clear_cache(self) -> None:
        self._cache.clear()

    async def check_plate(
        self,
        plate_persian: str,
        plate_latin: str,
        confidence: float = 0.0,
        force_refresh: bool = False,
    ) -> ApiCheckResult:
        """پلاک را به API سرور استعلام می‌کند و نتیجه را برمی‌گرداند.

        این تابع درخواست `POST` به آدرس `settings.check_url` (به‌صورت پیش‌فرض
        `http://myweb.web.com/api/plate/check`) ارسال می‌کند و پاسخ آن را
        به‌صورت یک `ApiCheckResult` تفسیر می‌کند.

        **محتوای بدنه درخواست (JSON):**
        - `plate`: شماره پلاک به‌صورت فارسی (مثلاً «۱۲ ب ۳۴۵ | ایران ۱۱»)
        - `plate_latin`: شماره پلاک به‌صورت لاتین (مثلاً «12B34511»)
        - `confidence`: میزان اطمینان مدل از خوانش پلاک (عدد بین 0 و 1)
        - `timestamp`: زمان ارسال به‌صورت ISO 8601

        **تفسیر پاسخ سرور (تبديل رنگ کادر):**
        - پاسخ متن ساده `"NO"` یا JSON دارای ارزش `NO` در فیلدهای
          `status` / `result` / `answer` / `response` / `decision` / `code`
          یا `allowed: false` ← وضعیت `REJECTED` (**کادر قرمز**)
        - هر پاسخ موفق (HTTP 2xx) غیر از `NO` (مثلاً `"YES"` یا
          `{"status": "OK", ...}`) ← وضعیت `APPROVED` (**کادر سبز**)
        - خطای شبکه یا timeout ← وضعیت `ERROR` (**کادر آبی + هشدار**)

        **نحوه مدیریت کش و تداخل:**
        - در صورت نبود `force_refresh`، نتایج قبلی از کش TTL (با دوره
          `api_cache_ttl_seconds`) بازگردانده می‌شوند تا درخواست تکراری
          ارسال نشود.
        - اگر درخواست هم‌زمان برای پلاک برابر در حال اجرا باشد، همان
          درخواست در حال اجرا بازگردانده می‌شود (تداخل درخواست حذف می‌شود).

        **مقایسه:** این تابع تنها مدخل API برای استعلام پلاک است و تمام
        مدیریت کش، تداخل و تفسیر پاسخ در آن متمرکز شده است.
        """
        plate_key = plate_latin or plate_persian
        if not force_refresh:
            cached = self.get_cached(plate_key)
            if cached is not None:
                return cached

        # اگر همین لحظه درخواست مشابهی در حال اجراست، منتظر همان نتیجه می‌مانیم
        existing_task = self._inflight.get(plate_key)
        if existing_task is not None and not existing_task.done():
            return await existing_task

        task = asyncio.create_task(
            self._do_check_plate(plate_persian, plate_latin, confidence)
        )
        self._inflight[plate_key] = task
        try:
            result = await task
            if result.status in (PlateCheckStatus.APPROVED, PlateCheckStatus.REJECTED):
                self._cache[plate_key] = (
                    time.monotonic() + self.settings.api_cache_ttl_seconds,
                    result,
                )
            return result
        finally:
            self._inflight.pop(plate_key, None)

    async def _do_check_plate(
        self,
        plate_persian: str,
        plate_latin: str,
        confidence: float,
    ) -> ApiCheckResult:
        payload = {
            "plate": plate_persian,
            "plate_latin": plate_latin,
            "confidence": round(confidence, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        t0 = time.perf_counter()
        try:
            client = self._get_client()
            if not self._require_https_base():
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return ApiCheckResult(
                    plate_persian=plate_persian,
                    plate_latin=plate_latin,
                    status=PlateCheckStatus.ERROR,
                    raw_response="",
                    latency_ms=latency_ms,
                    checked_at=datetime.now().strftime("%H:%M:%S"),
                    error_message="آدرس سرور باید از پروتکل HTTPS استفاده کند.",
                )
            resp = await client.post(self.settings.check_url, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if self._response_too_large(resp):
                return ApiCheckResult(
                    plate_persian=plate_persian,
                    plate_latin=plate_latin,
                    status=PlateCheckStatus.ERROR,
                    raw_response="",
                    latency_ms=latency_ms,
                    checked_at=datetime.now().strftime("%H:%M:%S"),
                    error_message="پاسخ سرور بیش از حد بزرگ است.",
                )
            return parse_server_plate_response(
                status_code=resp.status_code,
                response_text=resp.text,
                plate_persian=plate_persian,
                plate_latin=plate_latin,
                latency_ms=latency_ms,
            )
        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ApiCheckResult(
                plate_persian=plate_persian,
                plate_latin=plate_latin,
                status=PlateCheckStatus.ERROR,
                raw_response="",
                latency_ms=latency_ms,
                checked_at=datetime.now().strftime("%H:%M:%S"),
                error_message="پاسخی از سرور دریافت نشد (Timeout)",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning("API request error: %s", _safe_error_text(exc))
            return ApiCheckResult(
                plate_persian=plate_persian,
                plate_latin=plate_latin,
                status=PlateCheckStatus.ERROR,
                raw_response="",
                latency_ms=latency_ms,
                checked_at=datetime.now().strftime("%H:%M:%S"),
                error_message=_safe_error_text(exc),
            )

    async def submit_violation(
        self,
        plate_persian: str,
        plate_latin: str,
        violation_type: str,
        description: str,
        plate_crop_jpeg: bytes | None = None,
        evidence_image_jpeg: bytes | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        location_accuracy_m: float | None = None,
    ) -> ViolationSubmissionResult:
        """گزارش تخلف (همراه با تاریخ شمسی، ساعت، موقعیت GPS و تصاویر) را به سرور ارسال می‌کند.

        این تابع یک تخلف جدید را به سرور ارسال کرده و محلی نیز ثبت می‌کند،
        به‌گونه‌ای که در صورت قطعی اتصال، اطلاعات از دست نرود.

        **محتوای بدنه درخواست (JSON) به آدرس `settings.violation_url`:**
        - `id`: شناسه‌ی منحصربه‌فرد تخلف (مولتی‌کالم: `v-YYYYmmddHHMMSS-6هکساخالی`)
        - `plate` / `plate_latin`: پلاک به‌صورت فارسی و لاتین
        - `violation_type`: نوع تخلف
        - `description`: توضیحات تخلف
        - `timestamp`: زمان ارسال به‌صورت ISO 8601
        - `jalali_date` / `jalali_time` / `jalali_datetime` / `jalali_weekday`:
          تاریخ و ساعت به‌صورت جلالی (شمسی)
        - `latitude` / `longitude` / `location_accuracy_m`: موقعیت GPS
        - `has_image` / `has_evidence_image`: نشانه‌ی وجود تصویر
        - `image_base64` / `evidence_image_base64`: تصویر برش پلاک و تصویر
          مستندات تخلف به‌صورت base64 (در صورت وجود)

        **اقدام‌ها و نتایج:**
        - در صورت پاسخ HTTP 2xx ← `success=True` (همگام‌سازی با سرور)
        - در صورت خطای سرور (کد غیر 2xx) ← `success=False`، اما رکورد **همان‌حظ
          محلی ذخیره می‌شود** و قابل ارسال‌مجدد است.
        - در صورت قطعی اتصال ← `success=False`، با ذخیره‌ی محلی و هشدار
          مناسب. در تمام موارد `saved_locally` نشان‌دهنده‌ی ثبت محلی است.

        نتایج در یک `ViolationSubmissionResult` قابل‌تست برمی‌گردند.
        """
        now_ir = now_iran_datetime()
        jalali_date = format_jalali_date(now_ir, persian_digits=False)
        jalali_time = format_jalali_time(now_ir, persian_digits=False)
        jalali_weekday = format_jalali_verbose_date(now_ir, persian_digits=True)
        jalali_datetime = f"{jalali_date} {jalali_time}"

        record_id = f"v-{now_ir.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        iso_ts = now_ir.astimezone(timezone.utc).isoformat()
        local_time = f"{jalali_date} - {jalali_time}"

        plate_img_file = save_violation_image_bytes(f"{record_id}_plate.jpg", plate_crop_jpeg)
        evidence_img_file = save_violation_image_bytes(
            f"{record_id}_evidence.jpg",
            evidence_image_jpeg,
        )

        payload: dict[str, Any] = {
            "id": record_id,
            "plate": plate_persian,
            "plate_latin": plate_latin,
            "violation_type": violation_type,
            "description": description.strip(),
            "timestamp": iso_ts,
            "jalali_date": jalali_date,
            "jalali_time": jalali_time,
            "jalali_datetime": jalali_datetime,
            "jalali_weekday": jalali_weekday,
            "latitude": latitude,
            "longitude": longitude,
            "location_accuracy_m": location_accuracy_m,
            "has_image": plate_crop_jpeg is not None,
            "has_evidence_image": evidence_image_jpeg is not None,
        }
        if plate_crop_jpeg:
            payload["image_base64"] = base64.b64encode(plate_crop_jpeg).decode("ascii")
        if evidence_image_jpeg:
            payload["evidence_image_base64"] = base64.b64encode(evidence_image_jpeg).decode("ascii")

        synced = False
        server_response = ""
        message = ""
        try:
            client = self._get_client()
            if self._require_https_base():
                resp = await client.post(self.settings.violation_url, json=payload)
                if self._response_too_large(resp):
                    server_response = ""
                    message = "پاسخ سرور بیش از حد بزرگ است؛ تخلف به‌صورت محلی ذخیره شد."
                else:
                    server_response = resp.text.strip()
                    if 200 <= resp.status_code < 300:
                        synced = True
                        message = "تخلف با موفقیت در سرور ثبت شد."
                    else:
                        message = f"خطای سرور هنگام ثبت تخلف (کد {resp.status_code}). نسخه محلی ذخیره شد."
            else:
                server_response = ""
                message = "آدرس سرور باید از پروتکل HTTPS استفاده کند؛ تخلف به‌صورت محلی ذخیره شد."
        except Exception as exc:
            logger.warning("Failed to submit violation online: %s", _safe_error_text(exc))
            server_response = _safe_error_text(exc)
            message = "اتصال به سرور برقرار نشد؛ تخلف به‌صورت محلی روی گوشی ذخیره شد."

        record = ViolationRecord(
            id=record_id,
            plate=plate_persian,
            plate_latin=plate_latin,
            violation_type=violation_type,
            description=description.strip(),
            timestamp=iso_ts,
            local_time=local_time,
            jalali_date=jalali_date,
            jalali_time=jalali_time,
            jalali_weekday=jalali_weekday,
            latitude=latitude,
            longitude=longitude,
            location_accuracy_m=location_accuracy_m,
            has_image=plate_crop_jpeg is not None,
            has_evidence_image=evidence_image_jpeg is not None,
            plate_image_file=plate_img_file,
            evidence_image_file=evidence_img_file,
            synced=synced,
            server_response=server_response,
        )
        saved_locally = self._append_local_log(asdict(record))

        return ViolationSubmissionResult(
            success=synced,
            message=message,
            saved_locally=saved_locally,
            server_response=server_response,
            record_id=record_id,
        )

    async def retry_violation_sync(self, record: ViolationRecord) -> ViolationSubmissionResult:
        """ارسال مجدد یک تخلف ذخیره‌شده محلی به سرور myweb.web.com.

        این تابع برای همگام‌سازی تخلفاتی است که پیش‌تر ناموفق ثبت شدند
        (مثلاً در زمان قطعی اتصال). محتویات درخواست همان سازوکار
        `submit_violation` را دارد، اما داده‌ها از رکورد محلی (`ViolationRecord`)
        و تصاویر ذخیره‌شده‌ی آن بازیابی می‌شوند.

        **اقدام‌ها و نتایج:**
        - در صورت پاسخ HTTP 2xx ← وضعیت همگام‌سازی در فایل محلی
          (`update_violation_sync_status`) به `synced=True` تغییر می‌کند و
          نتیجه با `success=True` برمی‌گردد.
        - در صورت خطای سرور یا قطعی اتصال ← `success=False` با message
          مناسب، اما رکورد محلی حفظ می‌شود تا ارسال مجدد موردنیاز باشد.
        """
        plate_bytes = read_violation_image_bytes(record.plate_image_file)
        evidence_bytes = read_violation_image_bytes(record.evidence_image_file)

        payload: dict[str, Any] = {
            "id": record.id,
            "plate": record.plate,
            "plate_latin": record.plate_latin,
            "violation_type": record.violation_type,
            "description": record.description,
            "timestamp": record.timestamp,
            "jalali_date": record.jalali_date,
            "jalali_time": record.jalali_time,
            "jalali_datetime": f"{record.jalali_date} {record.jalali_time}".strip(),
            "jalali_weekday": record.jalali_weekday,
            "latitude": record.latitude,
            "longitude": record.longitude,
            "location_accuracy_m": record.location_accuracy_m,
            "has_image": plate_bytes is not None,
            "has_evidence_image": evidence_bytes is not None,
        }
        if plate_bytes:
            payload["image_base64"] = base64.b64encode(plate_bytes).decode("ascii")
        if evidence_bytes:
            payload["evidence_image_base64"] = base64.b64encode(evidence_bytes).decode("ascii")

        try:
            client = self._get_client()
            if self._require_https_base():
                resp = await client.post(self.settings.violation_url, json=payload)
                if self._response_too_large(resp):
                    return ViolationSubmissionResult(
                        success=False,
                        message="پاسخ سرور بیش از حد بزرگ است.",
                        saved_locally=True,
                        server_response="",
                        record_id=record.id,
                    )
                server_response = resp.text.strip()
                if 200 <= resp.status_code < 300:
                    update_violation_sync_status(record.id, synced=True, server_response=server_response)
                    return ViolationSubmissionResult(
                        success=True,
                        message="تخلف با موفقیت به سرور ارسال و همگام‌سازی شد.",
                        saved_locally=True,
                        server_response=server_response,
                        record_id=record.id,
                    )
                return ViolationSubmissionResult(
                    success=False,
                    message=f"سرور پاسخ خطا داد (کد {resp.status_code}).",
                    saved_locally=True,
                    server_response=server_response,
                    record_id=record.id,
                )
            return ViolationSubmissionResult(
                success=False,
                message="آدرس سرور باید از پروتکل HTTPS استفاده کند.",
                saved_locally=True,
                server_response="",
                record_id=record.id,
            )
        except Exception as exc:
            logger.warning("Retry violation sync failed: %s", _safe_error_text(exc))
            return ViolationSubmissionResult(
                success=False,
                message="ارسال مجدد ناموفق بود.",
                saved_locally=True,
                server_response=_safe_error_text(exc),
                record_id=record.id,
            )

    @staticmethod
    def _append_local_log(record: dict[str, Any]) -> bool:
        try:
            log_file = get_violation_log_file()
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            logger.warning("Could not write local violation log: %s", exc)
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()


def _normalize_search_text(text: str) -> str:
    """نرمال‌سازی متن فارسی/انگلیسی و ارقام برای جستجوی بدون خطا در تخلفات."""
    normalized = to_latin_digits(text or "").strip().lower()
    return (
        normalized.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("‌", " ")
    )


def filter_violation_records(
    records: list[ViolationRecord],
    *,
    search_query: str = "",
    sync_filter: str = "all",
    violation_type: str = "",
) -> list[ViolationRecord]:
    """
    فیلتر و جستجوی لیست تخلفات ثبت‌شده.

    sync_filter:
      - 'all': همه تخلفات
      - 'synced': فقط ارسال‌شده‌ها به سرور
      - 'pending': فقط در انتظار ارسال (ذخیره‌شده محلی)
      - 'gps': فقط دارای موقعیت مکانی GPS
    """
    q = _normalize_search_text(search_query)
    vtype_filter = (violation_type or "").strip()

    result: list[ViolationRecord] = []
    for rec in records:
        if sync_filter == "synced" and not rec.synced:
            continue
        if sync_filter == "pending" and rec.synced:
            continue
        if sync_filter == "gps" and not rec.has_gps:
            continue
        if vtype_filter and vtype_filter != "همه" and rec.violation_type != vtype_filter:
            continue

        if q:
            haystack = _normalize_search_text(
                " ".join(
                    [
                        rec.plate,
                        rec.plate_latin,
                        rec.violation_type,
                        rec.description,
                        rec.jalali_date,
                        rec.jalali_time,
                        rec.jalali_weekday,
                        rec.formatted_gps if rec.has_gps else "",
                        rec.id,
                    ]
                )
            )
            # جستجوی چندکلمه‌ای: تمام کلمات واردشده باید در رکورد وجود داشته باشند
            if not all(token in haystack for token in q.split()):
                continue

        result.append(rec)

    return result
