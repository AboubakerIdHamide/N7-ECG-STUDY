# Étude de cas 3 — Jalon 2 : décisions et résultats

## Objectif

Remplacer les descripteurs calculés à la main par un CNN 1D qui apprend directement à partir des 1 250 mesures d’une fenêtre ECG. En complément, entraîner un autoencodeur à reconstruire un signal Holter propre après ajout de bruit simulé, puis vérifier si ce débruitage aide un CNN Holter à classer le patch B.

## Données et protocole

- **Holter A :** 8 400 fenêtres train (280 patients), 1 800 validation (60 patients), 1 800 test (60 patients). Chaque fenêtre dure 10 secondes à 125 Hz (1 250 mesures). Le CNN est entraîné sur train et choisi sur validation.
- **Patch B :** 300 fenêtres train (12 patients), 150 validation (6 patients), 1 050 test (42 patients). Le patch test n’est pas chargé.
- Aucun patient n’est partagé entre train et validation au sein de chaque appareil. Les signaux sont simulés.
- Le test Holter avait déjà été évalué au Jalon 1. Il n’a pas été relu au Jalon 2 et n’a servi à aucune décision ici ; cette réutilisation historique doit être signalée comme limite du protocole global.

La normalisation soustrait la moyenne et divise par l’écart-type **de chaque fenêtre**. L’entrée CNN a la forme `(nombre de fenêtres, 1250, 1)`. Les poids de classe sont calculés uniquement avec les étiquettes train Holter.

## CNN : comparaison sur validation Holter A

Chaque modèle utilise Adam, une entropie croisée catégorielle creuse, des poids de classe, la graine 42, des lots de 128 et au plus 12 époques. Les architectures respectent les limites estimées de 2 Mo et 100 ms par fenêtre. Le classement se fait selon le F1 macro de validation.

| Architecture | Exactitude val. | F1 macro val. | Rappel classes 0–4 | Poids estimés | Latence CPU médiane |
|---|---:|---:|---|---:|---:|
| 3 blocs, noyau 7, moyenne globale | 0,748 | 0,779 | 0,559 / 0,979 / 0,942 / 0,894 / 0,900 | 0,082 Mo | 6,34 ms |
| 4 blocs, noyau 7, moyenne globale | 0,976 | 0,973 | 0,966 / 0,996 / 0,976 / 0,980 / 0,989 | 0,197 Mo | 7,75 ms |
| **4 blocs, noyau 11, moyenne globale** | **0,993** | **0,993** | **0,991 / 1,000 / 0,990 / 1,000 / 0,989** | **0,303 Mo** | **7,97 ms** |
| 4 blocs, noyau 7, aplatissement | 0,977 | 0,977 | 0,984 / 0,992 / 0,938 / 1,000 / 0,961 | 0,835 Mo | 7,96 ms |

Le modèle retenu est le CNN à **4 blocs, noyau 11, moyenne globale**. Son fichier Keras occupe 0,970 Mo ; la latence médiane locale est 7,97 ms par fenêtre. Le notebook ne consigne ni le modèle exact du processeur ni la mémoire disponible au moment de cette mesure : la latence est indicative et peut varier selon la machine.

Le MLP du Jalon 1 sur les mêmes données Holter A obtient F1 macro 0,946 pour la référence pondérée et 0,970 pour le modèle retenu selon le meilleur F1. Le CNN atteint 0,993 sur cette validation. Il reçoit les mesures brutes normalisées, alors que le MLP reçoit descripteurs construits à la main. Ces scores élevés sont propres à ce jeu simulé et ne constituent pas une validation clinique.

## Champ réceptif

Pour quatre blocs `Conv1D(k=7, stride=1)` suivis d’un max-pooling de taille 2, le champ réceptif théorique est **106 mesures**, soit `106/125 = 0,848 s`. À 70 battements/minute, un intervalle RR vaut environ `60/70 = 0,857 s` et deux RR valent 1,714 s. Le contexte local est donc juste inférieur à un RR et ne couvre pas deux RR. Ajouter des blocs ou employer des convolutions dilatées peut élargir le contexte. L’agrégation globale résume les activations dans toute la fenêtre, mais ne garantit pas à elle seule que les relations entre battements éloignés sont apprises.

## Autoencodeur et transfert vers Patch B

L’autoencodeur voit une entrée Holter artificiellement bruitée (variation de gain, dérive lente, composante secteur 50 Hz et bruit aléatoire) et apprend à reconstruire la fenêtre propre. Pour limiter le calcul CPU, un tirage reproductible de **1 200 fenêtres train et 300 validation** a été utilisé ; l’entraînement a duré 25 époques. La perte MSE finale de validation est **0,1176**.

Le CNN Holter retenu est ensuite gardé identique et évalué sur les 150 fenêtres de validation Patch B :

| Entrée du classifieur | F1 macro | Rappel classes 0–4 |
|---|---:|---|
| Patch B normalisé, sans autoencodeur | **0,836** | 0,821 / 0,833 / 1,000 / 0,778 / 0,760 |
| Patch B après autoencodeur Holter | 0,676 | 0,806 / 0,167 / 1,000 / 0,778 / 0,480 |

Dans cette expérience, le débruitage **réduit** le F1 macro de 0,160 ; il ne faut donc pas le présenter comme une amélioration. La validation Patch B est petite et ne contient que six exemples de classe 1, ce qui rend ses métriques par classe incertaines. Le bruit artificiel ne reproduit pas nécessairement le bruit réel de l’appareil.

## Limites et décision

Les expériences sont réalisées sur des signaux simulés ; aucune performance médicale ne peut en être déduite. Le CNN atteint la cible de 0,97 de F1 macro sur la validation Holter, mais cela ne démontre pas que la cible est atteinte sur un test indépendant. Le score Patch B sans débruitage (0,836) est inférieur à la cible système de 0,85, et le débruitage empirique détériore le résultat. Le modèle ne doit pas être déployé sur cette base.

Les comparaisons historiques CNN sur Patch B (F1 macro : 0,344 ; 0,549 ; 0,678 ; 0,238) sont exploratoires seulement : elles ne remplacent pas la comparaison principale CNN/MLP faite sur Holter A. Les résultats détaillés et le code exécuté sont conservés dans le notebook et le journal.
