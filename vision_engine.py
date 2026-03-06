# FICHIER: vision_engine.py
# ROLE: Moteur d'inférence en deux étages.
# Etage 1 = modèle de classe globale (humans, hand-signs, fictional, animals, vehicles)
# Etage 2 = matching de sous-type (peace, loser, modèle véhicule, etc.)

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_model.h5"
LABELS_PATH = BASE_DIR / "my_model_labels.json"
DATASET_DIR = BASE_DIR / "dataset"
METADATA_CSV = DATASET_DIR / "image_descriptions.csv"


# Modèle Etage 1: prédiction de la classe globale
model = tf.keras.models.load_model(str(MODEL_PATH))

# Modèle Etage 2: extracteur de features pour matching visuel de sous-type
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")


def _load_class_names():
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return sorted(
        [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    )


class_names = _load_class_names()


def _extract_feature_from_path(img_path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(str(img_path), target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = feature_extractor.predict(x, verbose=0)[0]
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def _build_subtype_refs():
    refs = {}
    if not METADATA_CSV.exists():
        return refs

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            file_name = (row.get("file_name") or "").strip()
            label = (row.get("label") or "").strip()

            if not class_name or not file_name or not label:
                continue

            img_path = DATASET_DIR / class_name / file_name
            if not img_path.exists():
                continue

            try:
                feat = _extract_feature_from_path(img_path)
            except Exception:
                continue

            refs.setdefault(class_name, []).append((label, feat))

    return refs


def _csv_mtime():
    try:
        return METADATA_CSV.stat().st_mtime
    except FileNotFoundError:
        return 0.0


SUBTYPE_REFS = _build_subtype_refs()
SUBTYPE_REFS_MTIME = _csv_mtime()
LABEL_DESCRIPTIONS = {}


def _norm_text(value: str) -> str:
    return (value or "").strip().lower()


def _ensure_subtype_refs_fresh():
    global SUBTYPE_REFS
    global SUBTYPE_REFS_MTIME
    global LABEL_DESCRIPTIONS

    current = _csv_mtime()
    if current != SUBTYPE_REFS_MTIME:
        SUBTYPE_REFS = _build_subtype_refs()
        LABEL_DESCRIPTIONS = _build_label_descriptions()
        SUBTYPE_REFS_MTIME = current


def _build_label_descriptions():
    descriptions = {}
    if not METADATA_CSV.exists():
        return descriptions

    with METADATA_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            class_name = _norm_text(row.get("class_name") or "")
            label = _norm_text(row.get("label") or "")
            desc = (row.get("description") or "").strip()
            if not class_name or not label or not desc:
                continue
            descriptions.setdefault(class_name, {})
            # Keep first non-empty description as canonical text
            descriptions[class_name].setdefault(label, desc)
    return descriptions


# Initialisation immédiate des descriptions au démarrage
LABEL_DESCRIPTIONS = _build_label_descriptions()


def _predict_subtype(img_path: str, predicted_class: str, k: int = 5):
    _ensure_subtype_refs_fresh()
    candidates = SUBTYPE_REFS.get(predicted_class, [])
    if not candidates:
        return None, 0.0

    q = _extract_feature_from_path(Path(img_path))
    scored = []
    for label, feat in candidates:
        sim = float(np.dot(q, feat))
        scored.append((sim, label))

    scored.sort(reverse=True, key=lambda t: t[0])
    top_k = scored[: max(1, min(k, len(scored)))]

    votes = Counter(lbl for _, lbl in top_k)
    best_label, best_count = votes.most_common(1)[0]
    confidence = best_count / len(top_k)
    return best_label, confidence


def get_label_description(class_name: str, label: str) -> str:
    _ensure_subtype_refs_fresh()
    return LABEL_DESCRIPTIONS.get(_norm_text(class_name), {}).get(_norm_text(label), "")


def analyze_image(img_path):
    """
    Retourne:
    - type_label: classe globale (etage 1)
    - type_confidence: confiance etage 1
    - detail_label: sous-type (etage 2)
    - detail_confidence: confiance etage 2
    """
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Etage 1
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    type_label = class_names[idx]
    type_confidence = float(preds[0][idx])

    # Etage 2
    detail_label, detail_confidence = _predict_subtype(img_path, type_label)
    if not detail_label:
        detail_label = type_label
        detail_confidence = type_confidence

    return type_label, type_confidence, detail_label, float(detail_confidence)


if __name__ == "__main__":
    test_file = BASE_DIR / "test.jpg"
    if test_file.exists():
        t, tc, d, dc = analyze_image(str(test_file))
        print(f"Type: {t} ({tc*100:.2f}%) | Label: {d} ({dc*100:.2f}%)")
    else:
        print("Put a test.jpg in this folder to test.")
