import numpy as np

from app_src.config import FS
from app_src.preprocessing.common import detect_qrs, validate_window

MAX_BEATS = 40


def build_beat_sequence(signal):
    """Créer une séquence complétée (40, 2) d’intervalles RR et d’amplitudes QRS."""
    values = validate_window(signal)
    peaks, _ = detect_qrs(values, FS)
    rr = np.diff(peaks).astype("float32") / FS
    amplitudes = values[peaks[1:]].astype("float32")
    sequence = np.zeros((MAX_BEATS, 2), dtype="float32")
    count = min(len(rr), MAX_BEATS)
    if count:
        sequence[:count] = np.column_stack([rr, amplitudes])[:count]
    return sequence, count


def normalize_beat_sequence(sequence, preprocessing):
    """Utiliser les statistiques calculées sur train et conserver le padding à zéro."""
    result = np.zeros_like(sequence, dtype="float32")
    mask = np.any(sequence != 0.0, axis=-1)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    std = np.asarray(preprocessing["std"], dtype=np.float32)
    result[mask] = (sequence[mask] - mean) / std
    return result[None, ...]
