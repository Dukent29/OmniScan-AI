import tensorflow as tf
from tensorflow.keras import layers, models
import csv
import json
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10

# Utiliser uniquement les catégories demandées (noms de dossiers dans dataset/)
SELECTED_CLASS_DIRS = ["humans", "hand-signs", "fictional", "animals", "vehicles"]
LABELS_PATH = BASE_DIR / "my_model_labels.json"
DESCRIPTIONS_CSV = DATA_DIR / "image_descriptions.csv"


def _load_description_stats(csv_path: Path, selected_classes: list[str]) -> tuple[int, int]:
    """
    Compte le nombre d'images sélectionnées avec description non vide.
    Ces descriptions ne sont pas utilisées directement dans le CNN (métadonnées de suivi).
    """
    if not csv_path.exists():
        return 0, 0

    total = 0
    described = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            class_name = (row.get("class_name") or "").strip()
            if class_name not in selected_classes:
                continue
            total += 1
            if (row.get("description") or "").strip():
                described += 1
    return described, total

# 1. CHARGEMENT DES DONNÉES (avec validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    class_names=SELECTED_CLASS_DIRS,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR),
    class_names=SELECTED_CLASS_DIRS,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Training categories: {class_names}")

described, total = _load_description_stats(DESCRIPTIONS_CSV, class_names)
if total > 0:
    print(f"Descriptions coverage: {described}/{total} images have descriptions in {DESCRIPTIONS_CSV}")
else:
    print(f"No descriptions file found for tracking at {DESCRIPTIONS_CSV}")

# 2. AUGMENTATION DES DONNÉES
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# 3. CONSTRUCTION DU MODÈLE
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

# IMPORTANT: on commence en gelant la base pré-entraînée
base_model.trainable = False

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax"),
])

# 4. PREMIÈRE PHASE: entraînement de la tête (5 epochs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("Phase 1: Training the top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS)

# 5. DEUXIÈME PHASE: fine-tuning (dé-geler la base)
print("Phase 2: Fine-tuning the whole brain...")
base_model.trainable = True

# On utilise un très petit learning rate pour préserver le savoir pré-entraîné
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS)

# 6. SAUVEGARDE
model.save("my_model.h5")
with open(LABELS_PATH, "w", encoding="utf-8") as fh:
    json.dump(class_names, fh, ensure_ascii=True, indent=2)
print(f"Training complete! Categories: {class_names}")
print(f"Saved label order to: {LABELS_PATH}")
