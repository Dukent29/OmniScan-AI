# Projet IA - Classification d'images (Big Data)

## 1. Objectif du projet
Ce projet permet de:
- analyser une image avec une IA,
- prédire une **classe principale** (ex: `humans`, `vehicles`, `hand-signs`, ...),
- estimer un **sous-type** (label) à partir des données annotées,
- enregistrer les résultats et l'historique dans **MongoDB**,
- corriger les erreurs via l'interface pour améliorer les prochains entraînements.

Le pipeline global est:
`Upload/Camera -> Analyse IA -> Sauvegarde MongoDB -> Affichage UI -> Correction -> Réentraînement`

---

## 2. Technologies utilisées et pourquoi

### Python
Langage principal pour l'IA, la logique applicative et les scripts de dataset.

### TensorFlow / Keras
- Entraînement du modèle de classification.
- Utilisation du transfert learning avec MobileNetV2 pour accélérer l'entraînement.

### Streamlit
- Interface web simple et rapide pour:
  - analyser des images,
  - visualiser l'historique,
  - corriger les prédictions,
  - ajouter des données d'entraînement.

### MongoDB (pymongo)
- Stockage des analyses et de l'historique.
- Traçabilité des corrections.

### icrawler
- Téléchargement automatisé d'images (Bing) pour construire le dataset.

---

## 3. Architecture IA (comment on l'utilise)

## Stage 1: Classe principale
Le modèle entraîné (`my_model.h5`) prédit une classe globale:
- humans
- hand-signs
- fictional
- animals
- vehicles

## Stage 2: Sous-type (label)
Le système compare l'image à des échantillons annotés dans `dataset/image_descriptions.csv` pour proposer un label (ex: `peace-sign`, `car`, `batman`, etc.).

Cela permet un comportement plus fin que la simple classe globale.

---

## 4. Structure du projet

```
my-ai-project/
├─ main.py                    # Interface Streamlit
├─ train_model.py             # Entraînement du modèle Stage 1
├─ vision_engine.py           # Analyse IA (Stage 1 + Stage 2)
├─ database_manager.py        # Opérations MongoDB
├─ mongo_connection.py        # Connexion MongoDB
├─ dataset_manager.py         # Ajout d'images + metadata dataset
├─ prepare_descriptions.py    # Synchronisation image_descriptions.csv
├─ download_images.py         # Téléchargement structuré d'images (icrawler)
├─ dataset/
│  └─ image_descriptions.csv  # class_name, file_name, label, description
└─ requirements.txt
```

---

## 5. Installation

## Prérequis
- Python 3.10+ recommandé
- MongoDB en local (ou URI distante)

## Installation des dépendances
Depuis le dossier du projet:

### Windows PowerShell
```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Lancer l'application

```bash
python -m streamlit run main.py
```

Dans l'interface:
- **Workspace (gauche)**: analyse + ajout de données
- **History (droite)**: table des prédictions, actions `Use`, `Delete`, `Correct & Add`

---

## 7. Préparer / enrichir le dataset

## Option A: Téléchargement automatique
Avant de lancer le téléchargement, créez le dossier racine `dataset`:

```bash
mkdir dataset
```

Les sous-dossiers (classes et sous-types) seront ensuite créés automatiquement par le script.

Puis lancez:

```bash
python download_images.py --reset --max-per-subtype 500
```

## Option B: Synchroniser le CSV après modifications manuelles
```bash
python prepare_descriptions.py
```

Le fichier `dataset/image_descriptions.csv` contient:
- `class_name`
- `file_name`
- `label`
- `description`

---

## 8. Entraîner le modèle

```bash
python train_model.py
```

Fichiers générés:
- `my_model.h5`
- `my_model_labels.json`

---

## 9. Base de données MongoDB

Base utilisée: `ImageAnalysisDB`

Collections principales:
- `ImageAnalysisResults`: résultats d'analyse de l'interface
- `TrainingDatasetEntries`: historique des ajouts de données d'entraînement

---

## 10. Boucle d'amélioration continue

1. Lancer l'app et analyser des images.
2. Si correct: bouton `Use` (échantillon validé).
3. Si faux: `Correct & Add` (type/label/description corrigés).
4. Réentraîner (`python train_model.py`).
5. Re-tester.

Cette boucle augmente progressivement la qualité du modèle.
