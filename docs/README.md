# Données du cas UC3

- `patch_B/` : tableaux `.npy` utilisés par les notebooks actuels et le fichier source `uc3_patchB.npz` fourni dans l’archive du cours. Le patch B contient 300 fenêtres train, 150 validation et 1 050 test (12, 6 et 42 patients).
- `holter_A/uc3_holterA.npz` : données Holter A récupérées dans l’archive : 8 400 train, 1 800 validation et 1 800 test (280, 60 et 60 patients). Le Jalon 1 demande ce jeu de données.
- `additional/uc3_inedits.npz` : 150 signaux non étiquetés avec leur fréquence cardiaque ; ne pas les utiliser comme données d’entraînement supervisé.

Toutes les fenêtres contiennent 1 250 mesures, soit 10 secondes à 125 Hz. Les signaux du sujet sont simulés et ne permettent aucune conclusion médicale. Les fichiers originaux de l’archive générale restent dans `../DOCS/Use Cases.rar`.
