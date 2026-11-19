"""سرویس دریافت موقعیت مکانی (GPS) مبتنی بر flet-geolocator.

این modul منطق کامل دریافت، سقوط نرم و نمایش موقعیت مکانی را از فایل اصلی
`main.py` استخراج کرده و آن را در یک کلاس متمرکز و قابل‌تست نگه می‌دارد.
هر تغییر در منطق دریافت GPS تنها در یک مکان انجام می‌شود.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class GpsService:
    """سرویس غیرهمزمان دریافت مختصات جغرافیایی گوشی با سقوط‌های نرم."""

    def __init__(
        self,
        page: Any,
        geolocator: Any | None,
        ftg: Any,
        state: Any,
    ) -> None:
        self.page = page
        self.geolocator = geolocator
        self.ftg = ftg
        self.state = state
        # ارجاع به صفحه ثبت تخلف (بعد از ساخته شدن آن در main.py تنظیم می‌شود)
        self.panel: Any | None = None

    async def fetch_current_gps(
        self, *, request_if_needed: bool = True
    ) -> tuple[tuple[float, float, float | None] | None, str]:
        """مختصات جغرافیایی فعلی گوشی (lat, lon, accuracy_m) را بدون قفل کردن رابط کاربری دریافت می‌کند.

        در صورت عدم دریافت موقعیت لحظه‌ای، اگر موقعیت کش‌شده قبلی وجود داشته باشد،
        آن را بی‌صدا برمی‌گرداند تا کاربر با خطای hardcore مواجه نشود.
        """
        if self.geolocator is None or self.ftg is None:
            return self.state.last_gps, "سرویس GPS در این محیط در دسترس نیست."

        if self.page.platform == self.ftg.PagePlatform.LINUX and not self.page.web:
            return self.state.last_gps, "سنسور GPS روی نسخه اندروید گوشی فعال است."

        try:
            service_on = await asyncio.wait_for(
                self.geolocator.is_location_service_enabled(), timeout=5.0
            )
            if not service_on:
                return self.state.last_gps, "سرویس موقعیت مکانی (Location/GPS) گوشی خاموش است."

            perm = await asyncio.wait_for(
                self.geolocator.get_permission_status(), timeout=5.0
            )
            granted_states = (
                self.ftg.GeolocatorPermissionStatus.WHILE_IN_USE,
                self.ftg.GeolocatorPermissionStatus.ALWAYS,
            )
            if perm not in granted_states:
                if not request_if_needed or perm == self.ftg.GeolocatorPermissionStatus.DENIED_FOREVER:
                    return self.state.last_gps, "مجوز دسترسی به موقعیت مکانی (GPS) صادر نشده است."
                perm = await asyncio.wait_for(self.geolocator.request_permission(), timeout=15.0)
                if perm not in granted_states:
                    return self.state.last_gps, "دسترسی به GPS توسط کاربر تأیید نشد."

            # مرحله اول: دریافت فوری آخرین موقعیت کش‌شده دستگاه (کمتر از ۵۰ میلی‌ثانیه)
            cached_pos = None
            if not self.page.web:
                try:
                    cached_pos = await asyncio.wait_for(
                        self.geolocator.get_last_known_position(), timeout=3.0
                    )
                except Exception:
                    cached_pos = None
            if (
                cached_pos is not None
                and cached_pos.latitude is not None
                and cached_pos.longitude is not None
            ):
                self.state.last_gps = (
                    float(cached_pos.latitude),
                    float(cached_pos.longitude),
                    float(cached_pos.accuracy) if cached_pos.accuracy else None,
                )
                self.state.last_gps_ts = time.monotonic()
                if self.state.screen_mode == "violation":
                    self._update_panel_from_state()

            # مرحله دوم: دریافت موقعیت دقیق و لحظه‌ای از GPS با زمان کافی
            # (تثبیت سیگنال GPS سرد می‌تواند در بسیاری از گوشی‌ها بیشتر از ۱۰ ثانیه طول بکشد)
            pos = await asyncio.wait_for(
                self.geolocator.get_current_position(
                    configuration=self.ftg.GeolocatorConfiguration(
                        accuracy=self.ftg.GeolocatorPositionAccuracy.HIGH,
                        time_limit=self.ftg.Duration(seconds=20),
                    )
                ),
                timeout=25.0,
            )
            if pos is not None and pos.latitude is not None and pos.longitude is not None:
                coords = (
                    float(pos.latitude),
                    float(pos.longitude),
                    float(pos.accuracy) if pos.accuracy else None,
                )
                self.state.last_gps = coords
                self.state.last_gps_ts = time.monotonic()
                return coords, ""

            # در صورت عدم دریافت موقعیت لحظه‌ای، با حفظ بدون هشدار خطا،
            # به آخرین موقعیت کش‌شده قبلی (اگر موجود باشد) برمی‌گردیم.
            if self.state.last_gps is not None:
                return self.state.last_gps, ""
            return None, "مختصات دقیقی از ماهواره GPS دریافت نشد (در فضای باز مجدد تلاش کنید)."
        except TimeoutError:
            # سقوط نرم: در صورت وجود آخرین موقعیت کش‌شده، بدون نمایش خطای timeout آن را نشان می‌دهیم
            if self.state.last_gps is not None:
                return self.state.last_gps, ""
            return None, "مختصات دقیق لحظه‌ای دریافت نشد؛ در فضای باز و با سیگنال GPS قوی مجدد تلاش کنید."
        except Exception as exc:
            logger.warning("GPS fetch error: %s", exc)
            if self.state.last_gps is not None:
                return self.state.last_gps, ""
            # نمایش پیام‌های دوست‌داشتنی فارسی به‌جای متن خطای سرویس (مثل «Location request timed out»)
            msg = str(exc).lower()
            if "timed out" in msg or "timeout" in msg:
                return None, "مختصات دقیق لحظه‌ای دریافت نشد؛ در فضای باز و با سیگنال GPS قوی مجدد تلاش کنید."
            if "permission" in msg or "denied" in msg:
                return None, "مجوز دسترسی به موقعیت مکانی (GPS) صادر نشده است."
            return None, "خطا در دریافت موقعیت مکانی (GPS). در فضای باز مجدد تلاش کنید."

    def _update_panel_from_state(self) -> None:
        """اگر صفحه در حالت ثبت تخلف است، آخرین موقعیت کش‌شده را در کارت GPS نمایش می‌دهد."""
        if self.panel is None:
            return
        try:
            self.panel.set_gps_location(*self.state.last_gps)
            self.panel.update()
        except Exception:
            pass

    async def refresh_panel_gps(self, panel: Any) -> None:
        """مختصات جدید را دریافت و در کارت موقعیت مکانی صفحه ثبت تخلف به‌روزرسانی می‌کند."""
        coords, err = await self.fetch_current_gps(request_if_needed=True)
        if self.state.screen_mode == "violation":
            if coords is not None:
                panel.set_gps_location(coords[0], coords[1], coords[2])
            else:
                panel.set_gps_location(None, None, None, error_note=err)
            try:
                panel.update()
            except Exception:
                pass
