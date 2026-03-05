import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "image_descriptions.csv"
SELECTED_CLASS_DIRS = ["humans", "hand-signs", "fictional", "animals", "vehicles"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_existing() -> dict[tuple[str, str], str]:
    if not CSV_PATH.exists():
        return {}
    existing: dict[tuple[str, str], dict[str, str]] = {}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = ((row.get("class_name") or "").strip(), (row.get("file_name") or "").strip())
            existing[key] = {
                "label": (row.get("label") or "").strip(),
                "description": (row.get("description") or "").strip(),
            }
    return existing


def build_rows(existing: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_name in SELECTED_CLASS_DIRS:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel_path = path.relative_to(class_dir).as_posix()
            key = (class_name, rel_path)
            default_label = rel_path.split("/", 1)[0] if "/" in rel_path else ""
            rows.append(
                {
                    "class_name": class_name,
                    "file_name": rel_path,
                    "label": existing.get(key, {}).get("label", default_label),
                    "description": existing.get(key, {}).get("description", ""),
                }
            )
    return rows


def main() -> None:
    existing = load_existing()
    rows = build_rows(existing)

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_name", "file_name", "label", "description"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {CSV_PATH} with {len(rows)} image rows.")
    print("Fill the 'description' column, then run: python train_model.py")


if __name__ == "__main__":
    main()
