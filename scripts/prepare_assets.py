"""Préparer les petits assets de déploiement sans lire les données de test."""
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app_src.config import ROOT
from app_src.preprocessing.jalon1 import FEATURE_NAMES, extract_descriptors
from app_src.preprocessing.jalon4 import build_beat_sequence

PRACTICE = ROOT.parent / "PRACTICE"
DATA = PRACTICE / "data"
ASSETS = ROOT / "assets"
MODEL_DIR = ASSETS / "models"
DEMO_DIR = ASSETS / "demo_data"
FS = 125


def load_split(dataset_name, split):
    filename = {"holter_A": "uc3_holterA.npz", "patch_B": "uc3_patchB.npz"}[dataset_name]
    path = DATA / dataset_name / filename
    with np.load(path) as archive:
        result = {f"X_{split}": archive[f"X_{split}"].copy(),
                  f"classe_{split}": archive[f"classe_{split}"].copy()}
        for key in ("age", "sexe"):
            name = f"{key}_{split}"
            if name in archive.files:
                result[name] = archive[name].copy()
    return result


def train_and_export_jalon1():
    """Reconstruire le MLP retenu, car aucun fichier .keras n’avait été enregistré."""
    train = load_split("holter_A", "train")
    val = load_split("holter_A", "val")

    # Ces statistiques sont ajustées uniquement sur les fenêtres du train.
    raw_train = np.vstack([
        extract_descriptors(x, age, sex)
        for x, age, sex in zip(train["X_train"], train["age_train"], train["sexe_train"])
    ])
    raw_val = np.vstack([
        extract_descriptors(x, age, sex)
        for x, age, sex in zip(val["X_val"], val["age_val"], val["sexe_val"])
    ])
    medians = np.nanmedian(raw_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    raw_train = np.where(np.isfinite(raw_train), raw_train, medians)
    raw_val = np.where(np.isfinite(raw_val), raw_val, medians)
    mean = raw_train.mean(axis=0)
    scale = raw_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = ((raw_train - mean) / scale).astype("float32")
    x_val = ((raw_val - mean) / scale).astype("float32")

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(11,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ], name="jalon1_mlp_descripteurs")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    model.fit(x_train, train["classe_train"].astype("int32"),
              validation_data=(x_val, val["classe_val"].astype("int32")),
              epochs=60, batch_size=64, verbose=0)
    model.save(MODEL_DIR / "jalon1_mlp_descripteurs.keras")
    metadata = {
        "feature_names": FEATURE_NAMES,
        "median": medians.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "alert_threshold_fa": 0.1057,
        "note": "Scaler ajusté uniquement sur Holter A train. Descripteurs QRS reconstruits comme dans le notebook; starter code officiel absent.",
    }
    (ASSETS / "jalon1_preprocessing.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MLP du Jalon 1 entraîné sur le train Holter A et exporté.")


def export_jalon4_scaler():
    """Recalculer les statistiques RR/amplitude en n’utilisant que les lignes train."""
    train = load_split("holter_A", "train")
    sequences = np.stack([build_beat_sequence(x)[0] for x in train["X_train"]])
    mask = np.any(sequences != 0.0, axis=-1)
    real_rows = sequences[mask]
    metadata = {
        "mean": real_rows.mean(axis=0).tolist(),
        "std": (real_rows.std(axis=0) + 1e-6).tolist(),
        "note": "Statistiques calculées uniquement sur les lignes de battements réelles de Holter A train.",
    }
    (ASSETS / "jalon4_preprocessing.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Normalisation du Jalon 4 calculée sur le train :", metadata["mean"], metadata["std"])


def export_validation_examples():
    """Emballer cinq exemples validation par source, un par classe, sans données test."""
    holter = load_split("holter_A", "val")
    patch = load_split("patch_B", "val")

    def first_per_class(dataset):
        indices = []
        for class_id in range(5):
            found = np.flatnonzero(dataset["classe_val"] == class_id)
            if not len(found):
                raise ValueError(f"La validation ne contient pas la classe {class_id}.")
            indices.append(int(found[0]))
        return np.asarray(indices, dtype=int)

    h_indices, p_indices = first_per_class(holter), first_per_class(patch)
    np.savez_compressed(
        DEMO_DIR / "demo_examples.npz",
        holter_raw=holter["X_val"][h_indices].astype("float32"),
        holter_labels=holter["classe_val"][h_indices].astype("int64"),
        holter_age=holter["age_val"][h_indices].astype("float32"),
        holter_sex=holter["sexe_val"][h_indices].astype("float32"),
        holter_validation_indices=h_indices,
        patch_raw=patch["X_val"][p_indices].astype("float32"),
        patch_labels=patch["classe_val"][p_indices].astype("int64"),
        patch_validation_indices=p_indices,
    )
    print("Exemples de validation exportés :", h_indices.tolist(), p_indices.tolist())


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    train_and_export_jalon1()
    export_jalon4_scaler()
    export_validation_examples()
