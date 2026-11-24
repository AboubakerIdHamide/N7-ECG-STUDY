# Jalon 3 — page de décisions

## Question

Le CNN appris sur Holter A garde-t-il ses performances sur Patch B ? Vaut-il mieux le laisser tel quel, d’entraîner un CNN avec les rares patients patch, ou de garder ses motifs Holter et d’ajuster seulement la sortie ?

## Données et protocole

Holter A comporte 8 400 fenêtres train (280 patients) et 1 800 validation (60 patients). Patch B comporte 300 fenêtres train (12 patients), 150 validation (6 patients) et 1 050 test (42 patients). Toutes les fenêtres durent 10 secondes à 125 Hz. Les modèles sont réglés par validation croisée groupée sur les 12 patients patch train. Le jeu de test n’est chargé qu’après le choix de la stratégie finale.

Les modèles utilisent l’architecture de départ : quatre blocs Conv1D(k=7) + max-pooling, moyenne globale, représentation dense de 64 unités et sortie à cinq classes. La normalisation z-score est faite séparément pour chaque fenêtre.

Pour les résultats patch, quatre plis GroupKFold sont faits sur les 12 patients patch train. Chaque pli entraîne sur 9 patients et évalue sur 3 patients complètement tenus à l’écart. Les métriques ci-dessous rassemblent les prédictions hors-pli.

## Décalage observé entre appareils (patients train uniquement)

| Descripteur brut médian | Holter A | Patch B | Lecture prudente |
|---|---:|---:|---|
| Écart-type (mV) | 0,2847 | 0,2451 | amplitude patch plus faible |
| Crête-à-crête (mV) | 2,0259 | 1,4470 | amplitude patch plus faible |
| Variation absolue par point (mV) | 0,0763 | 0,0649 | variation locale plus faible dans cet échantillon |
| Part de puissance 20–62,5 Hz | 0,0486 | 0,0458 | proche avec ce descripteur |
| Part autour de 50 Hz | 0,0034 | 0,0032 | pas d’écart marqué mesuré ici |

Après z-score, moyenne et écart-type par fenêtre sont proches de 0 et 1 sur les deux appareils : la normalisation retire un décalage constant et un gain multiplicatif. Elle ne retire pas automatiquement les dérives variables dans le temps, le bruit ajouté, ni les changements de forme. Les mesures ci-dessus sont des résumés simples de données simulées ; elles ne décrivent pas toutes les différences morphologiques et ne permettent aucune conclusion médicale.

## Résultats actuels — Patch B, validation croisée par patient

| Méthode | Exactitude | F1 macro | Rappel classes 0–4 |
|---|---:|---:|---|
| CNN Holter, appliqué tel quel au patch | 0,600 | 0,621 | 0,395 / 0,565 / 1,000 / 0,379 / 1,000 |
| CNN entraîné sur Patch B seulement | 0,360 | 0,344 | 0,395 / 0,246 / 0,630 / 0,448 / 0,054 |
| Extraction de caractéristiques : sortie seule ajustée | 0,780 | 0,771 | 0,748 / 0,812 / 0,935 / 0,655 / 0,730 |
| Ajustement fin complet du CNN Holter | 0,823 | 0,823 | 0,739 / 0,870 / 0,935 / 0,862 / 0,838 |
| Holter entraîné avec augmentation patch, sans adaptation | 0,703 | 0,687 | 0,622 / 0,623 / 0,978 / 0,448 / 0,973 |
| **Augmentation Holter + ajustement fin complet** | **0,837** | **0,844** | **0,756 / 0,899 / 0,935 / 0,931 / 0,784** |

L’extraction de caractéristiques a obtenu des F1 macro de 0,653 ; 0,664 ; 0,845 ; 0,464 sur les quatre plis. Elle gèle les couches qui analysent le signal Holter et n’ajuste que la couche de sortie : **325 paramètres entraînables**.


L’ajustement fin repart du CNN Holter, mais autorise toutes les couches à apprendre avec un taux d’apprentissage faible (10⁻⁴) : **51 365 paramètres entraînables**. Ses F1 macro par pli sont 0,777 ; 0,647 ; 0,853 ; 0,497.

Pour l’augmentation, environ la moitié des fenêtres Holter est perturbée avec les quatre transformations du code de départ (gain, dérive lente, secteur 50 Hz, bruit musculaire), avant normalisation. Sans ajustement patch, elle passe le F1 macro de 0,621 à 0,687. Combinée à l’ajustement fin, elle atteint 0,844 ; les F1 par pli sont 0,741 ; 0,724 ; 0,836 ; 0,511. C’est le meilleur score provisoire actuel, mais l’écart entre plis et les seuls 12 patients imposent de rester prudent.

Le CNN entraîné sur patch seul est moins bon que le transfert Holter direct dans cette expérience. Cela ne prouve pas que tout apprentissage patch seul échoue : le résultat dépend de l’architecture, des paramètres et des seuls 12 patients disponibles.

## Résultats du modèle multitâche

La tête fréquence cardiaque prédit une valeur en bpm. La cible FC est standardisée dans chaque pli à partir des patients patch train de ce pli ; λ règle le poids de son erreur MSE par rapport à la perte de classification. Les résultats agrégés sont :

| λ FC | Exactitude rythme CV | F1 macro rythme CV | MAE FC | Rappels rythme classes 0–4 |
|---:|---:|---:|---:|---|
| 0 | 0,830 | 0,828 | 43,2 bpm | 0,756 / 0,870 / 0,935 / 0,862 / 0,838 |
| **0,1** | **0,833** | **0,832** | 18,6 bpm | 0,748 / 0,884 / 0,935 / 0,828 / 0,892 |
| 1 | 0,810 | 0,801 | **15,8 bpm** | 0,782 / 0,841 / 0,935 / 0,828 / 0,676 |

La tâche FC améliore nettement la qualité de sa propre prédiction lorsque λ augmente. λ=0,1 donne le meilleur F1 rythme du petit balayage, mais ne dépasse pas l’augmentation + ajustement fin (0,844). Avec λ=1, la FC est mieux estimée, tandis que le F1 rythme diminue : les deux objectifs peuvent entrer en compétition.

## Vérifications théoriques

Le test numérique de z(2,5x+3) et z(x) donne un écart maximal de 1,43×10⁻⁵, dû à la petite constante numérique du z-score. Le bruit variable donne un écart moyen de 0,0899 après normalisation : le z-score retire un gain positif et un décalage constants, mais pas les perturbations qui changent au cours du temps.

Le comptage par le code confirme **325 paramètres entraînables** en extraction (seule la sortie rythme est dégelée) et **51 365** en ajustement fin complet.

## Choix final et test unique

La stratégie choisie avant le test est l’augmentation Holter + ajustement fin (F1 macro CV 0,844). Réentraînée sur les 12 patients Patch B train, elle obtient sur les 42 patients test une exactitude de **0,870**, un F1 macro de **0,803** et des rappels par classe de **0,824 / 0,854 / 0,972 / 1,000 / 0,875**. Le score test est inférieur au score CV, ce qui illustre la difficulté de généraliser avec peu de patients et un décalage d’appareil. Il n’a pas servi à modifier le modèle.

Lors d’une vérification technique préalable du fichier, le résumé numérique de la cible fréquence cardiaque du jeu test a été consulté pour vérifier son contenu. Cette information n’a pas servi à l’apprentissage, au choix de λ ou au choix du modèle ; la métrique de classification du test a été calculée une seule fois après la sélection.

## Recommandation d’annotation

Annoter en priorité **12 nouveaux patients Patch B** (ou de nouveaux enregistrements équivalents), afin de doubler le nombre actuel de patients patch train. Les sélectionner par diversité de qualité du signal et par désaccord/incertitude des modèles, puis vérifier que les rythmes actuellement moins fréquents sont représentés. Ne pas prélever ces nouveaux exemples dans le jeu de test réservé. Cette quantité est une recommandation pratique, pas un optimum calculé par les données simulées.

Les signaux sont simulés ; aucune conclusion médicale n’est possible.
