"""موتور سبک و خالص پایتون برای چسباندن حروف فارسی (Reshaping) و چینش راست‌به‌چپ (BiDi).

بدون هیچ کتابخانه خارجی (بدون نیاز به arabic-reshaper یا python-bidi/Rust):
- پشتیبانی کامل از تمام حروف فارسی (پ، چ، ژ، گ، ک، ی، ه، آ، ئ، ؤ، ...)
- پشتیبانی از لیگچرهای «لا / لآ / لأ / لإ»
- پشتیبانی از نیم‌فاصله (ZWNJ) برای کلماتی مانند «ثبت‌شده» و «می‌شود»
- حفظ جهت چپ‌به‌راست اعداد، تاریخ‌ها (1405/08/23)، ساعت‌ها (15:42:18) و کدهای لاتین (12B34511)
  درون جملات راست‌به‌چپ فارسی هنگام رسم با Pillow (`ImageFont.Layout.BASIC`).
"""

from __future__ import annotations

import re

# نگاشت حروف فارسی/عربی به چهار شکل یونیکد: (جدا، پایانی، آغازی، میانی)
# برای حروفی که فقط از راست می‌چسبند، دو شکل آخر None است.
_FORMS: dict[str, tuple[str, str | None, str | None, str | None]] = {
    "\u0621": ("\ufe80", None, None, None),          # ء
    "\u0622": ("\ufe81", "\ufe82", None, None),      # آ
    "\u0623": ("\ufe83", "\ufe84", None, None),      # أ
    "\u0624": ("\ufe85", "\ufe86", None, None),      # ؤ
    "\u0625": ("\ufe87", "\ufe88", None, None),      # إ
    "\u0626": ("\ufe89", "\ufe8a", "\ufe8b", "\ufe8c"),  # ئ
    "\u0627": ("\ufe8d", "\ufe8e", None, None),      # ا
    "\u0628": ("\ufe8f", "\ufe90", "\ufe91", "\ufe92"),  # ب
    "\u0629": ("\ufe93", "\ufe94", None, None),      # ة
    "\u062a": ("\ufe95", "\ufe96", "\ufe97", "\ufe98"),  # ت
    "\u062b": ("\ufe99", "\ufe9a", "\ufe9b", "\ufe9c"),  # ث
    "\u062c": ("\ufe9d", "\ufe9e", "\ufe9f", "\ufea0"),  # ج
    "\u062d": ("\ufea1", "\ufea2", "\ufea3", "\ufea4"),  # ح
    "\u062e": ("\ufea5", "\ufea6", "\ufea7", "\ufea8"),  # خ
    "\u062f": ("\ufea9", "\ufeaa", None, None),      # د
    "\u0630": ("\ufeab", "\ufeac", None, None),      # ذ
    "\u0631": ("\ufead", "\ufeae", None, None),      # ر
    "\u0632": ("\ufeaf", "\ufeb0", None, None),      # ز
    "\u0633": ("\ufeb1", "\ufeb2", "\ufeb3", "\ufeb4"),  # س
    "\u0634": ("\ufeb5", "\ufeb6", "\ufeb7", "\ufeb8"),  # ش
    "\u0635": ("\ufeb9", "\ufeba", "\ufebb", "\ufebc"),  # ص
    "\u0636": ("\ufebd", "\ufebe", "\ufebf", "\ufec0"),  # ض
    "\u0637": ("\ufec1", "\ufec2", "\ufec3", "\ufec4"),  # ط
    "\u0638": ("\ufec5", "\ufec6", "\ufec7", "\ufec8"),  # ظ
    "\u0639": ("\ufec9", "\ufeca", "\ufecb", "\ufecc"),  # ع
    "\u063a": ("\ufecd", "\ufece", "\ufecf", "\ufed0"),  # غ
    "\u0640": ("\u0640", "\u0640", "\u0640", "\u0640"),  # ـ (کشیده)
    "\u0641": ("\ufed1", "\ufed2", "\ufed3", "\ufed4"),  # ف
    "\u0642": ("\ufed5", "\ufed6", "\ufed7", "\ufed8"),  # ق
    "\u0643": ("\ufed9", "\ufeda", "\ufedb", "\ufedc"),  # ك (عربی)
    "\u0644": ("\ufedd", "\ufede", "\ufedf", "\ufee0"),  # ل
    "\u0645": ("\ufee1", "\ufee2", "\ufee3", "\ufee4"),  # م
    "\u0646": ("\ufee5", "\ufee6", "\ufee7", "\ufee8"),  # ن
    "\u0647": ("\ufee9", "\ufeea", "\ufeeb", "\ufeec"),  # ه
    "\u0648": ("\ufeed", "\ufeee", None, None),      # و
    "\u0649": ("\ufeef", "\ufef0", None, None),      # ى
    "\u064a": ("\ufef1", "\ufef2", "\ufef3", "\ufef4"),  # ي (عربی)
    # حروف ویژه فارسی (Presentation Forms-A)
    "\u067e": ("\ufb56", "\ufb57", "\ufb58", "\ufb59"),  # پ
    "\u0686": ("\ufb7a", "\ufb7b", "\ufb7c", "\ufb7d"),  # چ
    "\u0698": ("\ufb8a", "\ufb8b", None, None),      # ژ
    "\u06a9": ("\ufb8e", "\ufb8f", "\ufb90", "\ufb91"),  # ک (فارسی)
    "\u06af": ("\ufb92", "\ufb93", "\ufb94", "\ufb95"),  # گ
    "\u06cc": ("\ufbfc", "\ufbfd", "\ufbfe", "\ufbff"),  # ی (فارسی)
    "\u06c0": ("\ufba4", "\ufba5", None, None),      # ۀ
}

# لیگچرهای لام + الف: (شکل جدا، شکل پایانی متصل به قبل)
_LAM_ALEF_LIGATURES: dict[str, tuple[str, str]] = {
    "\u0622": ("\ufef5", "\ufef6"),  # لآ
    "\u0623": ("\ufef7", "\ufef8"),  # لأ
    "\u0625": ("\ufef9", "\ufefa"),  # لإ
    "\u0627": ("\ufefb", "\ufefc"),  # لا
}

# حرکات اعراب (تأثیری در اتصال حروف قبل و بعد ندارند)
_DIACRITICS = set(
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0670"
)

_MIRROR_MAP = str.maketrans("()[]{}<>«»", ")(][}{><»«")

# توکن‌های چپ‌به‌راست (اعداد فارسی/لاتین، تاریخ، ساعت، کلمات انگلیسی، آدرس وب)
_LTR_RUN_RE = re.compile(
    r"[A-Za-z0-9\u06f0-\u06f9\u0660-\u0669]"
    r"(?:[A-Za-z0-9\u06f0-\u06f9\u0660-\u0669._:/\-+%@]*[A-Za-z0-9\u06f0-\u06f9\u0660-\u0669])?"
)


def _connects_to_next(ch: str) -> bool:
    forms = _FORMS.get(ch)
    return forms is not None and forms[2] is not None


def _connects_to_prev(ch: str) -> bool:
    forms = _FORMS.get(ch)
    return forms is not None and forms[1] is not None


def _has_rtl_chars(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (0x0600 <= code <= 0x06FF) or (0xFB50 <= code <= 0xFEFF):
            if not (0x06F0 <= code <= 0x06F9 or 0x0660 <= code <= 0x0669):
                return True
    return False


def reshape_persian(text: str) -> str:
    """حروف فارسی/عربی را بر اساس موقعیت در کلمه به شکل صحیح یونیکد متصل می‌کند."""
    if not text:
        return ""

    chars = list(text)
    n = len(chars)
    out: list[str] = []
    i = 0

    while i < n:
        ch = chars[i]
        # نیم‌فاصله فقط اتصال را می‌شکند و در خروجی چاپی حذف می‌شود
        if ch in ("\u200c", "\u200d", "\u200e", "\u200f"):
            i += 1
            continue

        if ch in _DIACRITICS:
            i += 1
            continue

        # بررسی اتصال به کاراکتر قبلیِ غیرحرکتی
        prev_ch = ""
        p = i - 1
        while p >= 0 and chars[p] in _DIACRITICS:
            p -= 1
        if p >= 0:
            prev_ch = chars[p]

        # بررسی اتصال به کاراکتر بعدیِ غیرحرکتی
        next_idx = i + 1
        while next_idx < n and chars[next_idx] in _DIACRITICS:
            next_idx += 1
        next_ch = chars[next_idx] if next_idx < n else ""

        # لیگچر «لا / لآ / لأ / لإ»
        if ch == "\u0644" and next_ch in _LAM_ALEF_LIGATURES:
            lig_isolated, lig_final = _LAM_ALEF_LIGATURES[next_ch]
            if _connects_to_next(prev_ch):
                out.append(lig_final)
            else:
                out.append(lig_isolated)
            i = next_idx + 1
            continue

        forms = _FORMS.get(ch)
        if forms is None:
            out.append(ch)
            i += 1
            continue

        conn_prev = _connects_to_next(prev_ch) and _connects_to_prev(ch)
        conn_next = _connects_to_next(ch) and _connects_to_prev(next_ch)

        isolated, final_f, initial_f, medial_f = forms
        if conn_prev and conn_next and medial_f:
            out.append(medial_f)
        elif conn_prev and final_f:
            out.append(final_f)
        elif conn_next and initial_f:
            out.append(initial_f)
        else:
            out.append(isolated)

        i += 1

    return "".join(out)


def to_visual_rtl(text: str) -> str:
    """یک خط متن فارسی/دوجهته را به ترتیب دیداری چپ‌به‌راست برای رسم در Pillow تبدیل می‌کند."""
    if not text:
        return ""

    # اگر خط کاملاً لاتین یا عددی باشد (بدون هیچ حرف فارسی)، دست‌نخورده بماند
    if not _has_rtl_chars(text):
        return text

    reshaped = reshape_persian(text)

    # تفکیک خط به قطعات LTR (اعداد، تاریخ، کد لاتین) و RTL (کلمات فارسی و فاصله/علائم)
    segments: list[tuple[bool, str]] = []
    pos = 0
    for match in _LTR_RUN_RE.finditer(reshaped):
        start, end = match.span()
        if start > pos:
            segments.append((False, reshaped[pos:start]))
        segments.append((True, match.group(0)))
        pos = end
    if pos < len(reshaped):
        segments.append((False, reshaped[pos:]))

    visual_parts: list[str] = []
    for is_ltr, seg in reversed(segments):
        if is_ltr:
            visual_parts.append(seg)
        else:
            mirrored = seg.translate(_MIRROR_MAP)
            visual_parts.append(mirrored[::-1])

    return "".join(visual_parts)


def wrap_persian_lines(
    text: str,
    measure_fn,
    max_width: float,
    max_lines: int = 8,
) -> list[str]:
    """متن طولانی فارسی را بر اساس عرض پیکسلی (measure_fn) به خطوط مناسب می‌شکند.

    خروجی خطوط خام منطقی (قبل از to_visual_rtl) است تا هر خط جداگانه شکل‌دهی شود.
    """
    if not text:
        return []

    result_lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.strip().split()
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if measure_fn(to_visual_rtl(candidate)) <= max_width:
                current = candidate
            else:
                result_lines.append(current)
                current = word
                if len(result_lines) >= max_lines:
                    break
        if len(result_lines) < max_lines:
            result_lines.append(current)
        if len(result_lines) >= max_lines:
            break

    return result_lines[:max_lines]
