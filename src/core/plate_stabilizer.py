from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from config import (
    PERSIAN_TO_LATIN_LETTER,
    compose_iranian_plate,
    to_latin_digits,
    to_persian_digits,
)

DIGIT_SLOTS = (0, 1, 3, 4, 5, 6, 7)
LETTER_SLOT = 2


@dataclass(slots=True)
class _PlateObservation:
    chars: tuple[str, ...]
    confidence: float
    timestamp: float


@dataclass(slots=True)
class StabilizedPlateResult:
    """خروجی تثبیت‌شده الگوریتم رأی‌گیری چندفریمی (Multi-Frame Consensus)."""

    chars_persian: list[str]
    plate_persian: str
    plate_latin: str
    confidence: float
    votes_count: int
    window_size: int
    stability_score: float
    is_confirmed: bool
    is_glitch_ignored: bool = False

    def status_badge_text(self) -> str:
        """برچسب کوتاه فارسی برای نمایش وضعیت تثبیت پلاک روی صفحه."""
        votes_fa = to_persian_digits(str(self.votes_count))
        win_fa = to_persian_digits(str(self.window_size))
        pct_fa = to_persian_digits(str(int(round(self.stability_score * 100))))
        if self.is_confirmed:
            return f"تثبیت‌شده ({votes_fa}/{win_fa} فریم • {pct_fa}٪)"
        return f"در حال تثبیت ({votes_fa}/{win_fa} فریم)..."


def _normalize_plate_chars(chars_persian: list[str]) -> tuple[str, ...] | None:
    """نرمال‌سازی و اعتبارسنجی ساختار ۸ کاراکتری پلاک استاندارد ایران."""
    if not chars_persian or len(chars_persian) != 8:
        return None

    normalized: list[str] = []
    for idx, raw_ch in enumerate(chars_persian):
        ch = (raw_ch or "").strip()
        if not ch:
            return None
        if idx in DIGIT_SLOTS:
            d = to_latin_digits(ch)
            if len(d) != 1 or not d.isdigit():
                return None
            normalized.append(d)
        else:
            # جایگاه حرف فارسی (شاخص ۲)
            ch = ch.replace("ي", "ی").replace("ك", "ک")
            if ch not in PERSIAN_TO_LATIN_LETTER:
                return None
            normalized.append(ch)
    return tuple(normalized)


def _char_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """تعداد کاراکترهای منطبق در جایگاه‌های متناظر دو پلاک ۸ کاراکتری."""
    return sum(1 for x, y in zip(a, b) if x == y)


class MultiFramePlateStabilizer:
    """
    فیلتر تثبیت چندفریمی پلاک (Multi-Frame Consensus / Majority Voting).

    مزایا در شرایط واقعی جاده و دوربین موبایل:
    - حذف خطاهای تک‌فریمی OCR (مانند جابه‌جایی موقت رقم ۲ و ۳ یا حرف ب و پ)
      با رأی‌گیری وزن‌دار بر اساس اطمینان (Confidence-Weighted Majority Voting)
      در هر یک از ۸ جایگاه پلاک.
    - جلوگیری از پرش پلاک روی نمایشگر در اثر یک فریم تار یا لرزش دست.
    - تشخیص سریع تغییر خودرو (در صورت مشاهده ۲ فریم متوالی از خودروی جدید،
      بافر بدون تأخیر روی پلاک جدید قفل می‌شود).
    - جلوگیری از ارسال استعلام اشتباه به سرور در حالت «استعلام خودکار» تا پیش از
      تأیید چندفریمی (`is_confirmed=True`).
    """

    def __init__(
        self,
        window_size: int = 5,
        min_confirmations: int = 2,
        max_age_sec: float = 3.5,
        min_similarity: int = 5,
        challenger_switch_frames: int = 2,
        min_stability_score: float = 0.70,
    ) -> None:
        self.window_size = max(2, int(window_size))
        self.min_confirmations = max(1, min(int(min_confirmations), self.window_size))
        self.max_age_sec = max(0.5, float(max_age_sec))
        self.min_similarity = max(1, min(8, int(min_similarity)))
        self.challenger_switch_frames = max(1, int(challenger_switch_frames))
        self.min_stability_score = max(0.5, min(1.0, float(min_stability_score)))

        self._window: deque[_PlateObservation] = deque(maxlen=self.window_size)
        self._challenger: deque[_PlateObservation] = deque(maxlen=self.challenger_switch_frames)

    def reset(self) -> None:
        """پاک‌سازی کامل تاریخچه فریم‌های اخیر (مثلاً هنگام توقف یا تغییر صحنه)."""
        self._window.clear()
        self._challenger.clear()

    def _prune_expired(self, now: float) -> None:
        while self._window and (now - self._window[0].timestamp) > self.max_age_sec:
            self._window.popleft()
        while self._challenger and (now - self._challenger[0].timestamp) > self.max_age_sec:
            self._challenger.popleft()

    def _compute_consensus(self, glitch_ignored: bool = False) -> StabilizedPlateResult | None:
        if not self._window:
            return None

        total = len(self._window)
        consensus_chars: list[str] = []
        slot_agreements: list[float] = []

        for slot_idx in range(8):
            weighted_scores: dict[str, float] = {}
            counts: dict[str, int] = {}
            for rank, obs in enumerate(self._window):
                ch = obs.chars[slot_idx]
                # وزن‌دهی بر اساس اطمینان مدل + برتری ملایم برای فریم‌های تازه‌تر در حالت تساوی
                recency_factor = 1.0 + (0.04 * rank)
                w = max(0.15, float(obs.confidence)) * recency_factor
                weighted_scores[ch] = weighted_scores.get(ch, 0.0) + w
                counts[ch] = counts.get(ch, 0) + 1

            best_char = max(
                weighted_scores.keys(),
                key=lambda c: (weighted_scores[c], counts[c]),
            )
            consensus_chars.append(best_char)
            slot_agreements.append(counts[best_char] / total)

        stability_score = sum(slot_agreements) / 8.0
        mean_conf = sum(obs.confidence for obs in self._window) / total

        # شرط تأیید نهایی: حداقل تعداد فریم + میانگین هم‌رأیی کافی + نبود جایگاه کاملاً مردد
        min_slot_agreement = min(slot_agreements)
        is_confirmed = (
            total >= self.min_confirmations
            and stability_score >= self.min_stability_score
            and min_slot_agreement >= 0.5
        )

        plate_persian, plate_latin, _ = compose_iranian_plate(
            "".join(consensus_chars[0:2]),
            consensus_chars[2],
            "".join(consensus_chars[3:6]),
            "".join(consensus_chars[6:8]),
        )
        return StabilizedPlateResult(
            chars_persian=consensus_chars,
            plate_persian=plate_persian,
            plate_latin=plate_latin,
            confidence=mean_conf,
            votes_count=total,
            window_size=self.window_size,
            stability_score=stability_score,
            is_confirmed=is_confirmed,
            is_glitch_ignored=glitch_ignored,
        )

    def update(
        self,
        chars_persian: list[str],
        confidence: float,
        *,
        now: float | None = None,
    ) -> StabilizedPlateResult | None:
        """
        تغذیه خوانش یک فریم جدید به فیلتر رأی‌گیری و دریافت خروجی تثبیت‌شده.
        """
        ts = time.monotonic() if now is None else float(now)
        self._prune_expired(ts)

        norm = _normalize_plate_chars(chars_persian)
        if norm is None:
            return self._compute_consensus(glitch_ignored=bool(self._window))

        obs = _PlateObservation(
            chars=norm,
            confidence=max(0.0, min(1.0, float(confidence))),
            timestamp=ts,
        )

        if not self._window:
            self._window.append(obs)
            self._challenger.clear()
            return self._compute_consensus()

        current = self._compute_consensus()
        assert current is not None
        current_tuple = tuple(current.chars_persian)

        if _char_similarity(norm, current_tuple) >= self.min_similarity:
            # متعلق به همین خودرو است (حتی اگر ۱ تا ۳ کاراکتر در این فریم نویز داشته باشد)
            self._window.append(obs)
            self._challenger.clear()
            return self._compute_consensus()

        # پلاک جدید با خودروی فعلی متفاوت است -> بررسی در بافر رقیب (جلوگیری از نویز تک‌فریمی)
        if self._challenger and _char_similarity(norm, self._challenger[-1].chars) >= self.min_similarity:
            self._challenger.append(obs)
        else:
            self._challenger.clear()
            self._challenger.append(obs)

        if len(self._challenger) >= self.challenger_switch_frames:
            # خودروی جدید قطعی شد -> جایگزینی سریع بافر فعال با خودروی جدید
            self._window.clear()
            self._window.extend(self._challenger)
            self._challenger.clear()
            return self._compute_consensus()

        # فریم نامنطبق اول را فعلاً نادیده می‌گیریم تا نمایشگر پرش نداشته باشد
        return self._compute_consensus(glitch_ignored=True)
