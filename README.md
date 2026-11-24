# CardioPatch — démonstrateur des quatre jalons

Application pédagogique Streamlit pour explorer les modèles ECG développés dans les Jalons 1 à 4. L’interface, le chargement des modèles et le prétraitement sont séparés afin de pouvoir expliquer chaque composant.

> Les données fournies sont simulées. Cette application est une démonstration de cours, pas un outil médical.

## Lancer l’application

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L’application est conçue pour Python 3.12. Elle utilise les poids Keras et les petits exemples de validation déjà placés dans `assets/`.

## Ce que la démonstration montre

1. **Choisir un jalon.** Les Jalons 1, 2 et 4 utilisent un exemple de validation Holter A ; le Jalon 3 utilise Patch B car son modèle final a été adapté à cette source.
2. **Observer une fenêtre ECG de 10 secondes.** Chaque exemple contient 1 250 mesures à 125 Hz.
3. **Lancer une vraie prédiction avec le modèle sauvegardé.** Le résultat affiche la classe estimée et les cinq scores produits par le modèle.
4. **Explorer la représentation du jalon.** Le Jalon 1 affiche ses 11 descripteurs ; le Jalon 4 affiche les battements détectés sous forme RR/QRS. Le vrai label de l’exemple est caché dans un volet jusqu’à la comparaison.
5. **Voir l’alerte FA** pour les modèles des Jalons 1 et 4 avec leur seuil fixé pendant la validation.

Les exemples inclus sont de petites sélections des partitions de validation ; aucun signal du test n’est embarqué. Les métriques présentées sont les résultats consignés dans les travaux, et ne sont pas recalculées par l’application.

## Organisation du code

```text
app.py                         Point d’entrée Streamlit
app_src/config.py              Noms des rythmes, chemins, seuils et repères
app_src/data.py                Chargement des exemples de démonstration
app_src/preprocessing/
  common.py                    Validation du format et détecteur QRS partagé
  jalon1.py                    Calcul et normalisation des 11 descripteurs
  jalon4.py                    Séquence RR/QRS et masquage du padding
app_src/inference/predict.py   Chargement caché des modèles et prédictions
app_src/ui/
  home.py                      Présentation et protocole
  demo.py                      Sélection, courbe ECG et résultat
assets/models/                 Modèles Keras déployés
assets/experimental_models/    Modèles conservés pour les expériences, non chargés par l’app
assets/demo_data/              Exemples de validation uniquement
notebooks/Jalon1..Jalon4/      Notebooks français pour expliquer les expériences
docs/decisions/                Pages de décision des quatre jalons
docs/journals/                 Journaux d’expériences CSV
scripts/prepare_assets.py      Prépare les exemples et les paramètres train-only
```

## Modèles déployés

| Jalon | Entrée du modèle | Source de l’exemple | Traitement montré |
|---|---|---|---|
| 1 | MLP, 11 descripteurs | Holter A | Détection QRS, statistiques RR, énergie, âge et code sexe |
| 2 | CNN 1D, 1 250 points | Holter A | Normalisation de chaque fenêtre |
| 3 | CNN transféré, 1 250 points | Patch B | Normalisation de chaque fenêtre |
| 4 | GRU, jusqu’à 40 lignes de 2 valeurs | Holter A | Détection QRS, intervalle RR, amplitude QRS et masquage |

Le notebook Jalon 1 ne sauvegardait pas de fichier modèle. `scripts/prepare_assets.py` reconstruit son MLP retenu avec l’architecture, la perte, la graine et les 60 époques documentées, en utilisant uniquement le train Holter A. Le scaler est ajusté sur ce même train ; aucune donnée test n’est lue. Cette reconstruction atteint 0,967 de F1 macro sur validation, légèrement différent des 0,970 du notebook ; l’application le signale pendant la démonstration.

Le détecteur QRS des Jalons 1 et 4 est la reconstruction pédagogique utilisée dans les notebooks, car le code de démarrage officiel était absent des fichiers reçus. Cela est reflété dans les notebooks et dans le rapport.

## Préparer ou reconstruire les petits assets

À partir du dossier `PROJECT`, avec les données originales toujours présentes dans `../PRACTICE/data/` :

```bash
python scripts/prepare_assets.py
```

Cette commande entraîne une seule fois le petit MLP du Jalon 1 sur train, calcule les statistiques de prétraitement du Jalon 4 sur train et extrait cinq exemples par source depuis validation. Elle n’accède pas aux clés test.

## Déploiement Streamlit Community Cloud

Le dépôt Git doit contenir `app.py`, `requirements.txt`, `assets/`, et `app_src/`. Dans Streamlit Community Cloud, choisir le dépôt GitHub, la branche, puis `app.py` comme fichier d’entrée. Le dépôt actuel n’a pas encore de remote Git configuré ; les instructions de dépôt seront complétées dès que son URL est connue.
