# Étude de cas 3 — Jalon 1 : page de décisions

## Objectif et protocole

Classer cinq rythmes à partir de fenêtres ECG Holter A de 10 secondes (1 250 points à 125 Hz), puis régler un seuil d’alerte fibrillation atriale (FA, classe 1). Tous les choix ont été faits sur train/validation. Le résultat test est rapporté sans ajustement à partir de ce résultat ; la partition test a toutefois été consultée plusieurs fois, notamment lors de la correction du chargement et de la présente exécution complète. Les signaux sont simulés ; aucune conclusion médicale n’est possible.

## Données utilisées

Holter A : 8 400 fenêtres d’entraînement (280 patients), 1 800 fenêtres de validation (60 patients) et 1 800 fenêtres de test (60 patients). Train et validation ne partagent aucun patient. Les effectifs train par classe sont `[4348, 1209, 1018, 995, 830]` ; validation : `[892, 238, 292, 198, 180]`.

## Descripteurs et MLP de référence

Le signal est résumé par 11 valeurs reconstruites : huit statistiques des intervalles RR (moyenne, écart-type, médiane, minimum, maximum, étendue, RMSSD, pNN50), l’énergie du signal filtré, l’âge et le code sexe. Le MLP comporte 64 puis 32 neurones cachés, un dropout de 20 % et cinq sorties softmax. Sur la validation, le MLP de référence pondéré obtient exactitude **0,944**, F1 macro **0,946** et rappels par classe `[0,909 ; 0,958 ; 0,979 ; 0,995 ; 0,989]`. Aucun cas n’a moins de deux QRS détectés.

Le notebook général `uc3_demarrage.ipynb` n’était pas dans l’archive. Les noms et formules exacts des onze descripteurs de départ et le détecteur QRS sont donc une reconstruction pédagogique à partir de la fiche ; le résultat doit être présenté avec cette limite.

## Comparaison des pertes (validation)

| Perte | Exactitude | F1 macro | Rappel FA (classe 1) |
|---|---:|---:|---:|
| Entropie croisée simple | **0,972** | **0,970** | 0,937 |
| Entropie croisée pondérée | 0,944 | 0,946 | **0,958** |
| Focale (γ=2) | 0,969 | 0,968 | 0,937 |

**Décision :** retenir l’entropie croisée simple, qui a le meilleur F1 macro sur validation. La perte pondérée augmente légèrement le rappel FA, mais a un F1 macro inférieur ; les erreurs de classes ne sont donc pas résumées par l’exactitude seule. Les poids équilibrés calculés avec `N/(K × Nₖ)` sont `[0,386 ; 1,390 ; 1,650 ; 1,688 ; 2,024]`.

## Descripteurs supplémentaires et ablation

Les 12 valeurs supplémentaires couvrent irrégularité RR, forme QRS et énergie par bandes de fréquence. Avec ces familles complètes, F1 macro validation = **0,957**. En retirant l’irrégularité RR : **0,953** ; la forme QRS : **0,964** ; l’énergie par bande : **0,963**. Cette ablation utilise la perte pondérée pour les quatre variantes. Elle montre un léger gain en retirant la forme QRS ou l’énergie par bande dans cette expérience ; les différences restent petites et propres à cette validation.

## Seuil d’alerte FA et test final

Le seuil est choisi comme le 95e percentile des scores FA des fenêtres non-FA de validation : **0,105686**. Validation : spécificité **0,949** (1 483/1 562 fenêtres non-FA sans alerte), sensibilité **0,979** (233/238 FA alertées).

Au test Holter, avec modèle et seuil inchangés : exactitude à cinq classes **0,961**, F1 macro **0,959**, rappels `[0,978 ; 0,904 ; 0,990 ; 0,993 ; 0,925]`. Pour l’alerte FA : spécificité **0,949**, sensibilité **0,971**. Le rappel FA du classement à cinq classes est **0,904**, tandis que la sensibilité de l’alerte à seuil est **0,971** : ce sont deux décisions différentes. Le test respecte ici les deux cibles FA approximatives (95 % de spécificité et 95 % de sensibilité), mais le F1 macro reste sous la cible finale de 0,97. Aucun réglage n’a été fait sur le test.

Le contrôle de cohérence QRS–fréquence cardiaque a signalé 207/8 400 fenêtres train et 38/1 800 validation à examiner (écart >15 bpm ou >20 %). Ce sont des cas à inspecter, pas des échecs confirmés : les positions QRS de référence ne sont pas fournies.

## Réponses théoriques

Pour une vraie classe 1 avec probabilités `(0,50 ; 0,30 ; 0,10 ; 0,05 ; 0,05)`, l’entropie croisée vaut `−ln(0,30) = 1,204`. Le gradient par rapport aux logits avec softmax vaut `p − cible` : `[0,50 ; −0,70 ; 0,10 ; 0,05 ; 0,05]`.

## Conclusion à présenter

« Sur le Holter A, le MLP à descripteurs obtient un F1 macro test de 0,959. Nous avons retenu l’entropie croisée simple d’après la validation, puis fixé le seuil FA sur validation. Le seuil donne 0,971 de sensibilité à 0,949 de spécificité sur le test. Les résultats sont proches des cibles FA, mais le F1 macro reste sous 0,97. Les descripteurs sont reconstruits car le notebook général de départ manque, et les signaux sont simulés. »
