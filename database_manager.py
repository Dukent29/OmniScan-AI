# FICHIER: database_manager.py
# ROLE: Gère toute la communication MongoDB (sauvegarde, lecture, suppression).

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pymongo.errors import PyMongoError

from mongo_connection import get_database

COLLECTION_NAME = "ImageAnalysisResults"
TRAINING_COLLECTION_NAME = "TrainingDatasetEntries"


def _get_collection():
    db = get_database()
    return db[COLLECTION_NAME]


def _get_training_collection():
    db = get_database()
    return db[TRAINING_COLLECTION_NAME]


def save_analysis(
    file_name,
    file_size,
    type_label,
    type_confidence,
    detail_label=None,
    detail_confidence=None,
    detail_description=None,
    source_path=None,
):
    """
    Sauvegarde les résultats d'analyse dans MongoDB.
    """
    match_score = round(((float(type_confidence) + float(detail_confidence or type_confidence)) / 2.0), 6)
    document = {
        "Date": datetime.now(timezone.utc),
        "Name": str(file_name),
        "Size": int(file_size),
        "Analysis": {
            "SuccessRate": round(float(type_confidence), 6),
            "Type": type_label,
            "Label": detail_label if detail_label else type_label,
            "LabelConfidence": round(float(detail_confidence), 6) if detail_confidence is not None else None,
            "Description": str(detail_description) if detail_description is not None else "",
            "MatchScore": match_score,
        },
        "SourcePath": source_path,
    }

    result = _get_collection().insert_one(document)
    return result.inserted_id


def save_training_entry(file_name, file_size, class_name, label, description, source_path=None):
    """
    Sauvegarde un événement d'ajout au dataset d'entraînement dans MongoDB.
    """
    document = {
        "Date": datetime.now(timezone.utc),
        "Name": str(file_name),
        "Size": int(file_size),
        "Type": str(class_name),
        "Label": str(label) if label is not None else "",
        "Description": str(description) if description is not None else "",
        "SourcePath": source_path,
    }
    result = _get_training_collection().insert_one(document)
    return result.inserted_id


def get_all_records():
    """
    Retourne toutes les images analysées dans la base.
    Utilisé pour la section 'Historique' de l'application.
    """
    return list(_get_collection().find())


def delete_record(record_id):
    """
    Supprime un enregistrement via son identifiant MongoDB.
    """
    from bson.objectid import ObjectId

    _get_collection().delete_one({"_id": ObjectId(record_id)})

def analyze_and_save(image_path):
    """
    Lance l'analyse IA sur l'image puis sauvegarde le résultat dans MongoDB.
    """
    from vision_engine import analyze_image

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    type_label, type_confidence, detail_label, detail_confidence = analyze_image(image_path)
    print(
        f"Analysis result -> Type: {type_label}, SuccessRate: {type_confidence:.6f}, "
        f"Label: {detail_label}, LabelConfidence: {detail_confidence:.6f}"
    )
    inserted_id = save_analysis(
        path.name,
        path.stat().st_size,
        type_label,
        type_confidence,
        detail_label,
        detail_confidence,
        detail_description=None,
        source_path=str(path),
    )
    print(f"Successfully saved in ImageAnalysisDB.{COLLECTION_NAME}! ID: {inserted_id}")
    return inserted_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze images and save results to MongoDB.")
    parser.add_argument("--image", help="Image path to analyze and save.")
    parser.add_argument("--list", action="store_true", help="List total records in ImageAnalysisDB.")
    args = parser.parse_args()

    try:
        if args.image:
            analyze_and_save(args.image)

        if args.list or not args.image:
            records = get_all_records()
            print(f"Total records in DB: {len(records)}")
    except (PyMongoError, FileNotFoundError, RuntimeError) as exc:
        print(f"Database operation failed. Is MongoDB running? Error: {exc}")
