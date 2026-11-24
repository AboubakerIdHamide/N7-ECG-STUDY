# Données, protocole et limites — Jalon 2

- `../../data/holter_A/uc3_holterA.npz` : Holter A (8 400 train, 1 800 validation, 1 800 test ; 1 250 mesures par fenêtre). Train/validation ont servi au CNN et à l’autoencodeur ; le test n’a pas été chargé dans ce Jalon.
- `../../data/patch_B/uc3_patchB.npz` : patch B (300 train, 150 validation, 1 050 test ; mêmes 1 250 mesures). Le modèle CNN appris sur Holter et l’autoencodeur Holter sont évalués sur la validation patch ; le patch test n’est pas chargé.
- Les signaux sont simulés. Le bruit ajouté pour l’autoencodeur est lui aussi synthétique. Les résultats sont pédagogiques, pas médicaux.
- Pour tenir dans le temps de calcul local, l’autoencodeur utilise un tirage reproductible de 1 200 fenêtres Holter train et 300 validation, pendant 25 époques. Il s’agit d’une expérience de faisabilité, pas d’un entraînement exhaustif.
- Le score de transfert Patch B validation baisse de F1 macro 0,836 à 0,676 après débruitage. Le résultat négatif est conservé tel quel.
- Le test Holter avait été évalué au Jalon 1. Il n’a pas été consulté pour sélectionner le CNN Jalon 2 ; cette consultation historique est une limite du protocole global.
- Le notebook général `uc3_demarrage.ipynb` n’était pas présent dans l’archive UC3 extraite. Les scripts de démarrage par jalon restent dans `../../docs/starter_code/UC3_ECG/`.
