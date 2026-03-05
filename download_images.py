import argparse
import csv
import shutil
from pathlib import Path

from icrawler.builtin import BingImageCrawler

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "image_descriptions.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# class -> subtype -> search query
QUERY_MAP = {
    "fictional": {
        "batman": "batman character cosplay",
        "spiderman": "spiderman character cosplay",
        "wonder-woman": "wonder woman character cosplay",
        "superman": "superman character cosplay",
        "darth-vader": "darth vader character cosplay",
    },
    "vehicles": {
        "car": "car vehicle",
        "suv": "suv vehicle",
        "truck": "truck vehicle",
        "moto": "motorcycle motorbike",
        "boat": "boat ship watercraft",
    },
    "humans": {
        "portrait": "person portrait face",
        "full-body": "person full body standing",
        "group": "group of people",
        "child": "child portrait",
        "elderly": "elderly person portrait",
    },
    "animals": {
        "dog": "dog animal",
        "cat": "cat animal",
        "bunny": "rabbit bunny animal",
        "elephant": "elephant animal",
        "snake": "snake reptile animal",
        "chicken": "chicken bird animal",
    },
    "hand-signs": {
        "peace-sign": "hand peace sign gesture",
        "loser-sign": "hand loser sign gesture",
        "west-coast-sign": "west coast hand sign gesture",
        "east-coast-sign": "east coast hand sign gesture",
        "perfect-sign": "ok hand sign perfect gesture",
        "good-sign": "thumbs up hand gesture",
    },
}


def ensure_structure() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for class_name, subtypes in QUERY_MAP.items():
        for subtype in subtypes:
            (DATASET_DIR / class_name / subtype).mkdir(parents=True, exist_ok=True)


def reset_dataset() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    ensure_structure()


def crawl_images(max_per_subtype: int) -> None:
    for class_name, subtypes in QUERY_MAP.items():
        for subtype, query in subtypes.items():
            out_dir = DATASET_DIR / class_name / subtype
            print(f"[download] {class_name}/{subtype}: '{query}' (max {max_per_subtype})")
            crawler = BingImageCrawler(
                feeder_threads=1,
                parser_threads=2,
                downloader_threads=4,
                storage={"root_dir": str(out_dir)},
            )
            crawler.crawl(keyword=query, max_num=max_per_subtype)


def rebuild_descriptions_csv() -> None:
    rows = []
    for class_name in QUERY_MAP:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rel_path = img_path.relative_to(class_dir).as_posix()  # includes subtype folder
            subtype = rel_path.split("/", 1)[0] if "/" in rel_path else class_name
            rows.append(
                {
                    "class_name": class_name,
                    "file_name": rel_path,
                    "label": subtype,
                    "description": "",
                }
            )

    rows.sort(key=lambda r: (r["class_name"], r["label"], r["file_name"]))
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_name", "file_name", "label", "description"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] Wrote {len(rows)} rows to {CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download structured dataset with subfolders.")
    parser.add_argument(
        "--max-per-subtype",
        type=int,
        default=500,
        help="Maximum images to download per subtype folder (default: 500).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing dataset folder before downloading.",
    )
    args = parser.parse_args()

    if args.reset:
        print("[reset] Clearing existing dataset folder")
        reset_dataset()
    else:
        ensure_structure()

    crawl_images(max_per_subtype=args.max_per_subtype)
    rebuild_descriptions_csv()
    print("[next] Run: python train_model.py")


if __name__ == "__main__":
    main()
