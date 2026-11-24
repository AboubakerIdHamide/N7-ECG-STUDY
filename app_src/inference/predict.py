import json
from functools import lru_cache

import numpy as np
import tensorflow as tf

from app_src.config import ALERT_THRESHOLDS, MILESTONES
from app_src.preprocessing.common import normalize_raw_window, validate_window
from app_src.preprocessing.jalon1 import FEATURE_NAMES, transform_descriptors
from app_src.preprocessing.jalon4 import build_beat_sequence, normalize_beat_sequence


@lru_cache(maxsize=4)
def load_model(jalon_name):
    path = MILESTONES[jalon_name]["model_file"]
    if not path.exists():
        raise FileNotFoundError(f"Modèle absent : {path.name}. Exécutez scripts/prepare_assets.py.")
    return tf.keras.models.load_model(path, compile=False)


@lru_cache(maxsize=1)
def load_preprocessing(name):
    from app_src.config import ASSETS
    path = ASSETS / f"{name}_preprocessing.json"
    if not path.exists():
        raise FileNotFoundError(f"Prétraitement absent : {path.name}.")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_window(jalon_name, signal, age=50, sex_code=0):
    """Exécuter le modèle sauvegardé et renvoyer scores et éléments explicatifs."""
    signal = validate_window(signal)
    model = load_model(jalon_name)
    extra = {}

    if jalon_name.startswith("Jalon 1"):
        preprocessing = load_preprocessing("jalon1")
        model_input, raw_features = transform_descriptors(signal, age, sex_code, preprocessing)
        extra["features"] = list(zip(FEATURE_NAMES, raw_features.tolist()))
    elif jalon_name.startswith("Jalon 4"):
        preprocessing = load_preprocessing("jalon4")
        sequence, count = build_beat_sequence(signal)
        model_input = normalize_beat_sequence(sequence, preprocessing)
        extra["beats"] = sequence[:count]
    else:
        model_input = normalize_raw_window(signal)

    scores = model(model_input, training=False).numpy().reshape(-1)
    predicted_class = int(np.argmax(scores))
    threshold = ALERT_THRESHOLDS.get(jalon_name)
    alert = None if threshold is None else bool(scores[1] >= threshold)
    return {
        "scores": scores,
        "predicted_class": predicted_class,
        "threshold": threshold,
        "alert": alert,
        "extra": extra,
    }
