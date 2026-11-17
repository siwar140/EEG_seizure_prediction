# GAT-LSTM and LSTM autoencoders for CHB-MIT seizure prediction

Pipeline de prédiction des crises épileptiques sur le dataset CHB-MIT, à partir des
features SWT (stationary wavelet transform), avec deux autoencodeurs (LSTM classique
et GAT-LSTM spatio-temporel) et plusieurs baselines et contrôles.

Tout se trouve dans le notebook lstm_gat_autoencoder_EEG.ipynb : lecture BIDS de
CHB-MIT, labellisation, filtrage, rejet d'artefacts, extraction de features (mises en
cache), les modèles, les baselines et l'évaluation.

## Données et labels

- Lecture de <root>/sub-<patient>/sub-<patient>_scans.tsv (heures de début), des
  fichiers EDF et des annotations *_events.tsv. Les temps de crise sont lus même
  pour les enregistrements dont l'EDF est manquant ou ignoré, afin que l'exclusion de
  4 h autour de chaque crise soit toujours appliquée.
- 18 canaux bipolaires communs à tous les enregistrements (FP1-F7 ... CZ-PZ ; le
  doublon T8-P8 garde sa première copie). Les enregistrements à montage référentiel
  (chb12 runs 27-29) voient leurs canaux bipolaires reconstruits en A - B (la
  référence commune s'annule). Un enregistrement auquel il manque un des 18 canaux
  est ignoré et journalisé (voir l'audit des canaux).
- FIR passe-bande 0.5-45 Hz sur chaque enregistrement entier, puis fenêtres de 5 s
  sans chevauchement.
- Une fenêtre est rejetée si un canal dépasse 1000 uV, si un canal est plat
  (std < 0.5 uV) ou si elle contient des NaN.
- Crise « leading » : commence au moins 4 h après la fin de la crise précédente (les
  crises groupées sont écartées, leur EEG préictal chevaucherait les suites de la
  crise précédente).
- Préictal : fenêtre entièrement dans [onset - 35 min, onset - 5 min) d'une crise
  leading (30 min de SOP, 5 min d'horizon de prédiction).
- Interictal : fenêtre à au moins 4 h avant ou après chaque crise (leading ou non).
- Toute autre fenêtre est exclue.

## Features par fenêtre de 5 s

SWT db4, 7 niveaux, sur chaque fenêtre de 5 s de chacun des 18 canaux (1280
échantillons, multiple de 2^7, donc pas de padding). Bandes à 256 Hz : cA7 0-1,
cD7 1-2, cD6 2-4, cD5 4-8, cD4 8-16, cD3 16-32, cD2 32-64 Hz (cD1 64-128 Hz est
exclue, le filtre à 45 Hz la vide). La SWT n'est pas décimée, donc chaque bande garde
1280 coefficients alignés dans le temps. Comme dans v4, la RMS de chaque bande est
calculée sur des blocs de temps égaux et log1p est appliqué : swt_steps blocs par
fenêtre (par défaut 5, un par seconde) -> 18 canaux x 7 bandes x swt_steps = 630
features. GAT : chaque canal est un nœud avec 7 x swt_steps features.
swt_steps=1 donne 7 par nœud (126).

Cache : <feature_cache_dir>/<patient>_swt<steps>_features.npz.

## Modèles

Le notebook supporte l'autoencodeur LSTM sur features plates (method="lstm") et
un autoencodeur spatio-temporel Graph Attention + LSTM (method="gat").

Pour GAT, chaque dérivation bipolaire EEG est un nœud du graphe. Deux nœuds sont
connectés quand leurs dérivations partagent une électrode ; chaque nœud a aussi une
auto-boucle. Les features d'un nœud sont les énergies SWT de son canal (7 bandes x
swt_steps blocs temporels). Deux couches d'attention sur le graphe apprennent les
relations spatiales à chaque fenêtre de 5 s. Un LSTM modélise ensuite la séquence
temporelle de 12 fenêtres et reconstruit les features du graphe.

Les deux autoencodeurs sont entraînés uniquement sur des séquences interictales et
utilisent le même scoring par erreur de référence et la même évaluation par alarmes.

## Baselines et contrôles (--methods)

Sur les mêmes features, folds, mise à l'échelle et fenêtres scorées :

- pca : erreur de reconstruction avec --baseline-pca-dim composantes.
- mahalanobis : distance sous une covariance de Ledoit-Wolf.
- iforest : score d'anomalie Isolation Forest.
- timeofday : régression logistique sur l'heure seule (sin/cos de l'heure).
- position : régression logistique sur la position de la fenêtre dans la session.
- logreg / gbm : modèles supervisés sur les features EEG.
- posclock / logreg_posclock : position + heure, avec ou sans EEG.

## Modes

- pooled (par défaut) : UN modèle entraîné sur tous les patients ensemble, avec
  normalisation par patient (median / IQR de ses données interictales de
  développement), validation croisée groupée par patient pour le seuil, et un jeu
  caché (holdout) jamais utilisé.
- per-patient : un modèle par patient, évalué en leave-one-seizure-out.

## Évaluation

- Jeu caché : dernière crise utilisable + 20 % les plus récents du temps interictal.
- Leave-one-seizure-out sur les données de développement, avec blocs interleaved
  (par défaut) ou contiguës.
- Alarmes : moyenne mobile causale de 1 min du score ; alarme quand elle atteint le
  seuil, puis 30 min de réfractaire. Une crise est prédite si une alarme tombe dans
  sa période préictale ; les autres alarmes sont des fausses alarmes, comptées par
  heure interictale.
- Seuil choisi en validation croisée : sensibilité maximale avec au plus
  --max-fpr fausses alarmes par heure.
- Prédicteur aléatoire (Schelter et al. 2006) : alarmes de Poisson au même taux de
  fausses alarmes, probabilité de toucher une SOP de 30 min = 1 - exp(-FPR * SOP).

## Installation

    pip install -r requirements.txt

## Utilisation

    python lstm_autoencoder.py --root "C:\dataset chb\.nemar" --out results_ae
    python lstm_autoencoder.py --root "C:\dataset chb\.nemar" --out results_ae --holdout-patients chb05 chb10 chb20
    python lstm_autoencoder.py --root "C:\dataset chb\.nemar" --out results_ae --mode per-patient --patients chb01
    python lstm_autoencoder.py --root "C:\dataset chb\.nemar" --out results_ae --features-only
    python lstm_autoencoder.py --root "C:\dataset chb\.nemar" --out results_ae --methods lstm timeofday
    python lstm_autoencoder.py --check-gpu

Depuis le notebook, éditer la cellule Settings puis Run All.

## Configuration matérielle

- Keras entraîne l'autoencodeur LSTM sur GPU NVIDIA automatiquement quand TensorFlow
  le voit.
- L'extraction de features tourne toujours sur CPU (mise en cache après le premier
  run).
- use_gpu=False dans les settings force le CPU.
