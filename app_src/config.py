from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"
DEMO_DATA = ASSETS / "demo_data" / "demo_examples.npz"

FS = 125
WINDOW_SECONDS = 10
WINDOW_SAMPLES = FS * WINDOW_SECONDS

RHYTHMS = {
    0: "Rythme sinusal",
    1: "Fibrillation auriculaire (FA)",
    2: "Extrasystoles ventriculaires fréquentes",
    3: "Tachycardie sinusale",
    4: "Bloc AV du deuxième degré",
}

MILESTONES = {
    "Jalon 1 — MLP et descripteurs": {
        "model_file": MODELS / "jalon1_mlp_descripteurs.keras",
        "source": "Holter A",
        "input_description": "11 descripteurs calculés à partir du signal et des métadonnées patient.",
        "metrics": "Notebook : F1 macro validation 0,970 ; modèle reconstitué pour la démo : 0,967 sur validation.",
    },
    "Jalon 2 — CNN sur signal brut": {
        "model_file": MODELS / "jalon2_cnn_holter.keras",
        "source": "Holter A",
        "input_description": "Les 1 250 mesures temporelles d’une fenêtre de 10 secondes.",
        "metrics": "Validation Holter A : F1 macro 0,993.",
    },
    "Jalon 3 — Transfert Holter vers Patch": {
        "model_file": MODELS / "jalon3_cnn_transfer_patch.keras",
        "source": "Patch B",
        "input_description": "Les 1 250 mesures, après normalisation de la fenêtre.",
        "metrics": "Validation croisée : F1 macro 0,844. Test rapporté : 0,803 ; exactitude 0,870.",
    },
    "Jalon 4 — GRU sur suite de battements": {
        "model_file": MODELS / "jalon4_gru_beats.keras",
        "source": "Holter A",
        "input_description": "Jusqu’à 40 lignes : intervalle RR et amplitude QRS.",
        "metrics": "Validation : F1 macro 0,822. Test rapporté : 0,848.",
    },
}

ALERT_THRESHOLDS = {
    "Jalon 1 — MLP et descripteurs": 0.1057,
    "Jalon 4 — GRU sur suite de battements": 0.64198,
}

CLASS_COLORS = ["#15857d", "#e5684c", "#7160c8", "#e0a33b", "#4477aa"]
