# PROJET-IA
# ENCG SETTAT 
# KHADIJA HOUMIRI 
<img src="image7.png" style="height:540px;margin-right:393px"/>
# HLAL KAWTAR 
<img src="image7.png" style="height:540px;margin-right:393px"/>
# Explication détaillée du code

Ce code est un projet d'analyse de données financières en Python, divisé en **deux grandes parties** : une analyse statistique exploratoire, puis du clustering (apprentissage non supervisé).

---

## PARTIE 1 — Chargement & Exploration des données

### 1. Importation des bibliothèques
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```
Ces bibliothèques servent à :
- **numpy / pandas** → manipulation de données
- **matplotlib / seaborn** → visualisation
- **scipy.stats** → tests statistiques

---

### 2. Chargement des données
```python
with zipfile.ZipFile(data_path, 'r') as z:
    with z.open('Finance_data.csv') as f:
        df = pd.read_csv(io.BytesIO(f.read()))
```
Le fichier `Finance_data.csv` est **compressé dans un .zip** stocké sur Google Drive. On l'ouvre directement sans décompresser sur le disque.

---

### 3. Exploration statistique
```python
numerical_cols = df.select_dtypes(include=np.number)
```
On sélectionne uniquement les **colonnes numériques** pour les calculs.

| Calcul | Ce que ça mesure |
|---|---|
| `mean()` | Moyenne — valeur centrale |
| `median()` | Médiane — valeur du milieu |
| `mode()` | Mode — valeur la plus fréquente |
| `var()` | Variance — dispersion au carré |
| `std()` | Écart-type — dispersion standard |
| `max() - min()` | Étendue — amplitude des valeurs |

---

### 4. Visualisations
- **Histogrammes** → voir la distribution de chaque variable
- **Box plots** → détecter les valeurs aberrantes (outliers)
- **Heatmap de corrélation** → voir quelles variables évoluent ensemble

---

### 5. Tests statistiques
```python
probability = np.mean(sample_data > sample_data.mean())
```
Probabilité qu'une valeur dépasse la moyenne.

```python
z_scores = stats.zscore(sample_data)
```
Le **Z-score** mesure combien d'écarts-types une valeur est éloignée de la moyenne. Un Z-score > 3 indique souvent une anomalie.

```python
t_stat, p_value = stats.ttest_1samp(sample_data, sample_data.mean())
```
Le **test T de Student** vérifie si la moyenne réelle est significativement différente d'une valeur de référence. Ici c'est un test trivial (comparé à sa propre moyenne → p-value = 1.0).

---

## PARTIE 2 — Clustering (Apprentissage non supervisé)

### 6. Normalisation des données
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numerical_cols)
```
On **standardise** les données : chaque variable aura une moyenne de 0 et un écart-type de 1. C'est **indispensable** pour les algorithmes de clustering qui sont sensibles aux échelles.

---

### 7. Algorithme K-Medians (fait maison)

C'est une variante du célèbre **K-Means**, mais plus robuste aux outliers.

```
K-Means  → utilise la MOYENNE  + distance Euclidienne (L2)
K-Medians → utilise la MÉDIANE + distance Manhattan   (L1)
```

**Étapes de l'algorithme :**

```
1. Initialisation  → choisir k points au hasard comme centres
        ↓
2. Assignation     → chaque point rejoint le centre le plus proche (L1)
        ↓
3. Mise à jour     → recalculer les centres avec la MÉDIANE
        ↓
4. Répéter jusqu'à convergence (plus aucun point ne change de cluster)
```

**Évaluation avec le Silhouette Score :**
- Score entre -1 et +1
- **Proche de 1** → clusters bien séparés et cohérents ✅
- **Proche de 0** → clusters qui se chevauchent ⚠️
- **Négatif** → mauvaise assignation ❌

---

### 8. Clustering Hiérarchique (HAC)

```python
for method in ['ward', 'complete', 'average', 'single']:
    hac = AgglomerativeClustering(n_clusters=3, linkage=method)
```

Il teste **4 méthodes de liaison** pour fusionner les clusters :

| Méthode | Critère de fusion |
|---|---|
| `ward` | Minimise l'augmentation de variance |
| `complete` | Distance maximale entre deux points |
| `average` | Distance moyenne entre tous les points |
| `single` | Distance minimale entre deux points |

Le **dendrogramme** est un arbre visuel qui montre comment les groupes se fusionnent progressivement — tracé sur 50 points pour rester lisible.

---

### 9. Gaussian Mixture Model (GMM)

```python
GaussianMixture(n_components=3, covariance_type=cov_type)
```

Contrairement au K-Means (clusters sphériques rigides), le GMM suppose que les données suivent un **mélange de distributions gaussiennes**. Il teste 4 formes de covariance :

| Type | Forme des clusters |
|---|---|
| `full` | Ellipses libres (le plus flexible) |
| `tied` | Même ellipse pour tous |
| `diag` | Axes alignés, tailles différentes |
| `spherical` | Sphères (comme K-Means) |

Il utilise aussi **BIC et AIC** pour trouver le **nombre optimal de clusters** : on cherche le k qui minimise le BIC.

---

### 10. Affinity Propagation

```python
ap = AffinityPropagation(damping=0.9, max_iter=500)
```

Algorithme **entièrement automatique** : il trouve lui-même le nombre de clusters sans qu'on le fixe à l'avance. Chaque point peut devenir un "exemplaire" (représentant de son cluster). Le paramètre `damping=0.9` ralentit les mises à jour pour assurer la convergence.

---

## Résumé global

```
Finance_data.csv
      │
      ├── Statistiques descriptives (moyenne, variance, etc.)
      ├── Visualisations (histogrammes, boxplots, heatmap)
      ├── Tests statistiques (Z-score, T-test)
      │
      └── Clustering
           ├── K-Medians     → robuste aux outliers (L1)
           ├── HAC           → arbre hiérarchique
           ├── GMM           → modèle probabiliste
           └── Affinity Prop → k automatique
```

Le but final est de **segmenter les données financières en groupes homogènes** sans étiquettes prédéfinies, et de comparer les performances de plusieurs algorithmes via le Silhouette Score.
