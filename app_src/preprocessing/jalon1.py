import numpy as np

from app_src.config import FS
from app_src.preprocessing.common import detect_qrs, validate_window

FEATURE_NAMES = [
    "RR moyen (s)", "Écart-type RR (s)", "Médiane RR (s)", "RR minimum (s)",
    "RR maximum (s)", "Étendue RR (s)", "RMSSD (s)", "pNN50",
    "Énergie du signal filtré", "Âge (ans)", "Code sexe",
]


def extract_descriptors(signal, age, sex):
    """Recalculer les 11 descripteurs de base du notebook pour une fenêtre ECG."""
    values = validate_window(signal)
    peaks, filtered = detect_qrs(values, FS)
    rr = np.diff(peaks) / FS
    if len(rr) == 0:
        rr_stats = [np.nan] * 8
    else:
        changes = np.diff(rr)
        rmssd = np.sqrt(np.mean(changes ** 2)) if len(changes) else np.nan
        pnn50 = np.mean(np.abs(changes) > 0.05) if len(changes) else np.nan
        rr_stats = [
            np.mean(rr), np.std(rr), np.median(rr), np.min(rr), np.max(rr),
            np.ptp(rr), rmssd, pnn50,
        ]
    return np.asarray(rr_stats + [np.mean(filtered ** 2), float(age), float(sex)], dtype=np.float32)


def transform_descriptors(signal, age, sex, preprocessing):
    features = extract_descriptors(signal, age, sex)
    median = np.asarray(preprocessing["median"], dtype=np.float32)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["scale"], dtype=np.float32)
    features = np.where(np.isfinite(features), features, median)
    return ((features - mean) / scale)[None, :].astype("float32"), features
