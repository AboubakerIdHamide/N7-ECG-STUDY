# Jalon 4 — page de décisions

Les premiers résultats sont des essais exploratoires de trois époques ; ils sont suivis d’une comparaison affinée sur validation, puis de l’évaluation finale unique sur test. Les réglages et le seuil ont été fixés à partir de train/validation avant cette dernière évaluation.

## Question et protocole

Comparer le RNN simple, le LSTM brut, l’hybride CNN + LSTM/GRU (CNN suivi d’une couche récurrente) et le GRU sur suite de battements. Les fenêtres d’un même patient restent dans le même ensemble.

## Résultats provisoires — validation seulement

| Modèle | F1 macro val. | Exactitude val. | Rappels classes 0–4 | Paramètres | s/époque |
|---|---:|---:|---|---:|---:|
| RNN simple sur ECG brut | 0,186 | 0,199 | 0,109 / 0,210 / 0,483 / 0,232 / 0,133 | 373 | 4,58 |
| LSTM sur ECG brut | 0,205 | 0,330 | 0,433 / 0,609 / 0,212 / 0,000 / 0,006 | 1 237 | 19,79 |
| CNN seul | 0,413 | 0,443 | 0,304 / 0,088 / 0,938 / 0,611 / 0,617 | 10 885 | 8,98 |
| CNN + LSTM | 0,083 | 0,127 | 0,001 / 0,025 / 0,068 / 1,000 / 0,022 | 15 829 | 13,57 |
| GRU sur suite RR–QRS | 0,581 | 0,600 | 0,552 / 0,286 / 0,671 / 0,788 / 0,933 | 3 621 | 1,24 |

Le LSTM gagne légèrement en F1 macro sur le RNN brut, mais manque presque toutes les classes 3 et 4 ; il a 3,3 fois plus de paramètres et 4,3 fois plus de temps par époque. Le CNN + LSTM utilise le même extracteur convolutionnel que le CNN seul ; seul le traitement de la séquence de caractéristiques change. À trois époques, le CNN seul dépasse nettement l’hybride. Le GRU sur suite de battements obtient alors le meilleur F1 macro, mais son rappel de classe 1 reste bas. Le tableau suivant affine la comparaison des trois candidats principaux.

### Comparaison affinée sur validation — 12 époques maximum

| Modèle | F1 macro val. | Exactitude val. | Rappels classes 0–4 | Paramètres | s/époque |
|---|---:|---:|---|---:|---:|
| CNN seul | 0,595 | 0,576 | 0,434 / 0,697 / 0,866 / 0,682 / 0,533 | 10 885 | 10,44 |
| CNN + LSTM | 0,420 | 0,422 | 0,234 / 0,214 / 0,719 / 0,949 / 0,567 | 15 829 | 14,59 |
| GRU sur suite RR–QRS | 0,822 | 0,821 | 0,762 / 0,761 / 0,918 / 0,899 / 0,950 | 3 621 | 0,97 |

Les trois modèles ont atteint le plafond de 12 époques ; le meilleur hybride a été observé à l’époque 11. Le modèle hybride progresse de F1 macro 0,083 à 0,420 avec davantage d’apprentissage, ce qui montre que l’essai initial de trois époques était trop court pour l’interpréter. Il reste inférieur au CNN seul dans cette comparaison. Le GRU obtient ici le meilleur score et des rappels plus équilibrés. Ce résultat dépend toutefois du détecteur QRS reconstruit : le notebook de démarrage officiel qui devait le fournir est absent. Les essais utilisent une seule graine et la validation ; le choix a été arrêté avant de consulter les métriques test.

## Choix et justification

Choix retenu pour l’évaluation finale : GRU sur suite de battements. Motifs décidés sur validation : meilleur F1 macro affiné parmi les candidats, rappels par classe relativement équilibrés, 3 621 paramètres et 0,97 s/époque. La sensibilité FA à 95 % de spécificité reste sous la cible, mais les cibles système sont des objectifs de comparaison, pas une condition pour terminer ce jalon. Le détecteur QRS est reconstruit et les mesures proviennent d’une seule graine.

### Objectifs système mesurés sur validation

| Cible finale du sujet | Résultat validation | Lecture |
|---|---:|---|
| F1 macro ≥ 0,97 | 0,822 | Sous la cible de 0,148 |
| Sensibilité FA ≥ 0,95 pour spécificité 0,95 | 0,592 à seuil 0,64198 ; spécificité 0,950 | Sous la cible de sensibilité de 0,358 |
| Taille modèle ≤ 2 Mo | 73 104 octets (environ 0,070 Mio) | Sous la limite |
| Latence ≤ 100 ms/fenêtre | 1,114 ms réseau seul ; 2,699 ms avec QRS et préparation | Sous la limite sur ce CPU, mesure indicative seulement |

La sensibilité FA a été mesurée pour la classe 1 ; le seuil 0,64198 a été réglé avec les fenêtres non-FA de validation. La latence bout en bout a été mesurée sur 100 fenêtres validation avec le détecteur reconstruit. Le notebook ne consigne ni le modèle exact du processeur ni la mémoire disponible pendant les mesures ; ces temps sont indicatifs et ne garantissent pas la performance sur le matériel embarqué réel.

### Évaluation finale unique — test Holter A

| Mesure test | Résultat |
|---|---:|
| Fenêtres / patients | 1 800 / 60 |
| F1 macro / exactitude | 0,848 / 0,832 |
| Rappels classes 0–4 | 0,776 / 0,721 / 0,980 / 0,934 / 0,949 |
| Seuil FA fixé sur validation | 0,64198 |
| Sensibilité FA / spécificité FA | 0,554 / 0,951 |

Le test confirme que le GRU ne rejoint pas la cible de F1 macro 0,97 ni la sensibilité FA 0,95, même si sa spécificité test est 0,951. Aucun réglage n’est modifié après ce résultat. Limite importante : cette même partition Holter A avait déjà été rapportée dans le Jalon 1 ; elle n’est donc pas vierge à l’échelle de tout le projet, même si le modèle Jalon 4 et son seuil ont été choisis sur validation.

## Vérifications théoriques

Si le gradient est multiplié à chaque pas par 0,9, après 1 250 pas il reste 0,9^1250 ≈ 6,36 × 10⁻⁵⁸ ; avec 0,99, il reste 0,99^1250 ≈ 3,5 × 10⁻⁶. Dans les deux cas, l’influence d’un gradient venant du début de la séquence devient très faible, particulièrement avec 0,9. Un LSTM Keras de 32 unités et une variable d’entrée compte 4 × 32 × (1 + 32 + 1) = 4 352 paramètres, confirmé par Keras. Pour les battements, les zéros complètent chaque suite jusqu’à 40 positions ; `Masking(mask_value=0)` fait ignorer ces positions au GRU afin qu’elles ne soient pas traitées comme de vrais battements.

## Limites

Les signaux sont simulés et ne permettent aucune conclusion médicale.
