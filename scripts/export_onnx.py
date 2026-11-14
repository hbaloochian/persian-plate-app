#!/usr/bin/env python3
"""ابزار کمکی تبدیل مدل‌های PyTorch (.pt) مخزن Persian_Plate_Recognition به فرمت ONNX.

این اسکریپت فقط یک بار (روی کامپیوتر یا داخل GitHub Actions) اجرا می‌شود و هیچ
وابستگی به اپلیکیشن نهایی اندروید اضافه نمی‌کند.

روش اجرا:
    pip install ultralytics onnx onnxslim
    python scripts/export_onnx.py --download-from-github --output-dir src/weights
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE_REPO_URL = "https://github.com/shahabbai/Persian_Plate_Recognition.git"
DEFAULT_WEIGHT_NAMES = ("cars_best_v26.pt", "chars_best_v26.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تبدیل مدل‌های YOLO26 تشخیص پلاک فارسی به فرمت ONNX بهینه برای اندروید"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="مسیر پوشه حاوی فایل‌های .pt (در صورت دانلود قبلی مخزن Persian_Plate_Recognition)",
    )
    parser.add_argument(
        "--download-from-github",
        action="store_true",
        help="دانلود خودکار وزن‌های YOLO26 از مخزن shahabbai/Persian_Plate_Recognition",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/weights"),
        help="پوشه مقصد برای ذخیره فایل‌های .onnx (پیش‌فرض: src/weights)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="ابعاد ورودی مدل ONNX (پیش‌فرض 640؛ برای سرعت بیشتر روی موبایل 480 یا 320)",
    )
    return parser.parse_args()


def export_model(pt_path: Path, output_dir: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    if not pt_path.is_file():
        raise FileNotFoundError(f"فایل وزن پیدا نشد: {pt_path}")

    print(f"[export] Exporting {pt_path.name} -> ONNX (imgsz={imgsz}, end2end=True)...")
    model = YOLO(str(pt_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            end2end=True,
            simplify=True,
            opset=17,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / exported.name
    shutil.copy2(exported, destination)
    print(f"[export] Saved: {destination} ({destination.stat().st_size / (1024 * 1024):.1f} MB)")
    return destination


def main() -> None:
    args = parse_args()

    if args.download_from_github:
        with tempfile.TemporaryDirectory() as tmp:
            repo_dir = Path(tmp) / "Persian_Plate_Recognition"
            print(f"[git] Cloning {SOURCE_REPO_URL} (shallow)...")
            subprocess.run(
                ["git", "clone", "--depth", "1", SOURCE_REPO_URL, str(repo_dir)],
                check=True,
            )
            weights_dir = repo_dir / "weights"
            for name in DEFAULT_WEIGHT_NAMES:
                export_model(weights_dir / name, args.output_dir, args.imgsz)
    else:
        source_dir = args.source_dir or Path("weights")
        pt_files = sorted(source_dir.glob("*.pt"))
        if not pt_files:
            raise SystemExit(
                f"هیچ فایل .pt در '{source_dir}' پیدا نشد. "
                "از گزینه --download-from-github استفاده کنید."
            )
        for pt_file in pt_files:
            export_model(pt_file, args.output_dir, args.imgsz)

    print("\nتبدیل با موفقیت انجام شد. فایل‌های .onnx آماده ساخت APK هستند.")


if __name__ == "__main__":
    main()
