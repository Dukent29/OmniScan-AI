import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DESCRIPTIONS_CSV = DATASET_DIR / "image_descriptions.csv"
ALLOWED_TYPES = ["humans", "hand-signs", "fictional", "animals", "vehicles"]
CSV_FIELDS = ["class_name", "file_name", "label", "description"]


def _safe_stem(name: str) -> str:
    stem = Path(name).stem.strip().lower()
    stem = re.sub(r"[^a-z0-9_-]+", "-", stem)
    return stem or "image"


def _safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"


def _safe_dir(name: str) -> str:
    value = (name or "").strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unlabeled"


def _ensure_csv_header() -> None:
    if DESCRIPTIONS_CSV.exists():
        return
    DESCRIPTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DESCRIPTIONS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()


def _upsert_metadata(class_name: str, file_name: str, label: str, description: str) -> None:
    _ensure_csv_header()
    rows = []
    updated = False

    with DESCRIPTIONS_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized = {
                "class_name": row.get("class_name", ""),
                "file_name": row.get("file_name", ""),
                "label": row.get("label", ""),
                "description": row.get("description", ""),
            }
            if normalized["class_name"] == class_name and normalized["file_name"] == file_name:
                normalized["label"] = label
                normalized["description"] = description
                updated = True
            rows.append(normalized)

    if not updated:
        rows.append(
            {
                "class_name": class_name,
                "file_name": file_name,
                "label": label,
                "description": description,
            }
        )

    with DESCRIPTIONS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def add_dataset_entry(image_bytes: bytes, original_name: str, class_name: str, label: str, description: str) -> Path:
    if class_name not in ALLOWED_TYPES:
        raise ValueError(f"Invalid class type: {class_name}")

    class_dir = DATASET_DIR / class_name
    subtype_dir = class_dir / _safe_dir(label)
    subtype_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"{_safe_stem(original_name)}-{timestamp}{_safe_suffix(original_name)}"
    output_path = subtype_dir / file_name
    output_path.write_bytes(image_bytes)
    file_name_rel = output_path.relative_to(class_dir).as_posix()

    _upsert_metadata(
        class_name=class_name,
        file_name=file_name_rel,
        label=(label or "").strip(),
        description=(description or "").strip(),
    )
    return output_path


def add_dataset_entries(entries: Iterable[dict]) -> list[Path]:
    saved_paths: list[Path] = []
    for entry in entries:
        saved_paths.append(
            add_dataset_entry(
                image_bytes=entry["image_bytes"],
                original_name=entry["original_name"],
                class_name=entry["class_name"],
                label=entry.get("label", ""),
                description=entry.get("description", ""),
            )
        )
    return saved_paths
