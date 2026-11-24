from functools import lru_cache

import numpy as np

from app_src.config import DEMO_DATA


@lru_cache(maxsize=1)
def load_demo_examples():
    """Charger un petit ensemble d’exemples de validation, sans données de test."""
    if not DEMO_DATA.exists():
        raise FileNotFoundError("Exemples absents. Exécutez scripts/prepare_assets.py.")
    with np.load(DEMO_DATA) as data:
        return {key: data[key].copy() for key in data.files}
