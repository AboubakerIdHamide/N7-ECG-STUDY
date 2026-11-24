import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from app_src.config import FS, WINDOW_SAMPLES


def validate_window(signal):
    """Vérifier puis renvoyer une fenêtre ECG finie de 1 250 mesures."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size != WINDOW_SAMPLES:
        raise ValueError(f"Une fenêtre doit contenir exactement {WINDOW_SAMPLES} mesures.")
    if not np.isfinite(values).all():
        raise ValueError("Le signal contient des valeurs manquantes ou non finies.")
    return values


def normalize_raw_window(signal):
    """Appliquer la normalisation par fenêtre utilisée aux Jalons 2 et 3."""
    values = validate_window(signal)
    return ((values - values.mean()) / (values.std() + 1e-6))[None, :, None].astype("float32")


def detect_qrs(signal, fs=FS):
    """Reproduire le détecteur QRS reconstruit dans les notebooks des jalons."""
    values = validate_window(signal).astype(float)
    sos = butter(3, [5, 18], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, values)
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative ** 2
    smooth_n = max(1, round(0.12 * fs))
    smooth = np.convolve(energy, np.ones(smooth_n) / smooth_n, mode="same")
    median = np.median(smooth)
    mad = np.median(np.abs(smooth - median))
    threshold = median + 4 * mad
    candidates, _ = find_peaks(
        smooth,
        distance=round(0.25 * fs),
        height=threshold,
        prominence=max(4 * mad, np.finfo(float).eps),
    )
    radius = round(0.08 * fs)
    peaks = []
    for candidate in candidates:
        left = max(0, candidate - radius)
        right = min(len(filtered), candidate + radius + 1)
        peaks.append(left + int(np.argmax(np.abs(filtered[left:right]))))
    return np.asarray(peaks, dtype=int), filtered
