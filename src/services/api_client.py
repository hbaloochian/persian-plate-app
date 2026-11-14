"""کلاینت غیرهمزمان ارتباط با API سرور myweb.web.com با کش هوشمند و ثبت تخلف."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx

from config import AppSettings, get_writable_data_dir

logger = logging.getLogger(__name__)


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
                follow_redirects=True,
                headers={
                    "Accept": "application/json, text/plain;q=0.9",
                    "User-Agent": "PersianPlateScanner/1.0-Android",
                },
            )
        return self._client

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
        """پلاک را به API در آدرس myweb.web.com ارسال کرده و نتیجه را برمی‌گرداند."""
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
            resp = await client.post(self.settings.check_url, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000.0
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
            logger.warning("API request error: %s", exc)
            return ApiCheckResult(
                plate_persian=plate_persian,
                plate_latin=plate_latin,
                status=PlateCheckStatus.ERROR,
                raw_response="",
                latency_ms=latency_ms,
                checked_at=datetime.now().strftime("%H:%M:%S"),
                error_message=f"خطا در اتصال به سرور: {exc}",
            )

    async def submit_violation(
        self,
        plate_persian: str,
        plate_latin: str,
        violation_type: str,
        description: str,
        plate_crop_jpeg: bytes | None = None,
    ) -> ViolationSubmissionResult:
        """گزارش تخلف را به سرور myweb.web.com ارسال و هم‌زمان یک نسخه محلی ذخیره می‌کند."""
        record = {
            "plate": plate_persian,
            "plate_latin": plate_latin,
            "violation_type": violation_type,
            "description": description.strip(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_image": plate_crop_jpeg is not None,
        }
        saved_locally = self._append_local_log(record)

        payload = dict(record)
        if plate_crop_jpeg:
            payload["image_base64"] = base64.b64encode(plate_crop_jpeg).decode("ascii")

        try:
            client = self._get_client()
            resp = await client.post(self.settings.violation_url, json=payload)
            if 200 <= resp.status_code < 300:
                return ViolationSubmissionResult(
                    success=True,
                    message="تخلف با موفقیت در سرور ثبت شد.",
                    saved_locally=saved_locally,
                    server_response=resp.text.strip(),
                )
            return ViolationSubmissionResult(
                success=False,
                message=f"خطای سرور هنگام ثبت تخلف (کد {resp.status_code}). نسخه محلی ذخیره شد.",
                saved_locally=saved_locally,
                server_response=resp.text.strip(),
            )
        except Exception as exc:
            logger.warning("Failed to submit violation online: %s", exc)
            return ViolationSubmissionResult(
                success=False,
                message="اتصال به سرور برقرار نشد؛ تخلف به‌صورت محلی روی گوشی ذخیره شد.",
                saved_locally=saved_locally,
                server_response=str(exc),
            )

    @staticmethod
    def _append_local_log(record: dict[str, Any]) -> bool:
        try:
            log_file = get_writable_data_dir() / "violations_log.jsonl"
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except OSError as exc:
            logger.warning("Could not write local violation log: %s", exc)
            return False

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
