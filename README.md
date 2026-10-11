# GNN Airline Link Prediction

## 1. Contexte

Le transport aérien constitue un réseau complexe dans lequel les aéroports sont interconnectés par des routes aériennes. Ce réseau peut naturellement être représenté sous forme de graphe :

- **Nœuds** : représentent les aéroports.
- **Arêtes** : représentent les routes aériennes reliant les aéroports.
- **Caractéristiques des nœuds** : décrivent les aéroports, notamment leur localisation géographique, leur pays et la population de leur ville.

Cette représentation permet d'étudier les caractéristiques individuelles des aéroports ainsi que leurs relations et leur position dans le réseau.

## 2. Problématique

Les données décrivant un réseau aérien peuvent être incomplètes : certaines routes existantes peuvent être absentes des données disponibles, tandis que certaines connexions potentielles n'existent pas.

La problématique de ce projet est donc la suivante :

> **Comment exploiter simultanément les caractéristiques des aéroports et la structure du réseau aérien afin de prédire l'existence de routes entre des paires d'aéroports dont la connexion n'est pas connue ?**

Cette problématique correspond à une tâche de **Link Prediction**, ou prédiction de liens, sur un graphe.

L'objectif n'est pas de classifier individuellement les aéroports, mais de déterminer si une relation entre deux nœuds est susceptible d'exister.

## 3. Approche proposée

Le projet repose sur une approche de prédiction de liens utilisant les **Graph Neural Networks (GNN)**.

### Pipeline général

```text
       Données des aéroports
                 |
                 v
       Construction du graphe
                 |
                 v
     Prétraitement des features
                 |
                 v
    Séparation train / validation / test
                 |
        +--------+--------+
        |                 |
        v                 v
     Baselines         Modèles GNN
  Adamic-Adar      GCN / GraphSAGE / GAT
       MLP           GAE / VGAE
        |                 |
        +--------+--------+
                 |
                 v
       Prédiction des liens
                 |
                 v
       Évaluation (AUC, AP)
                 |
                 v
      Ablation du meilleur modèle
                 |
                 v
       Analyse des liens suspects
```

Une partie des connexions connues est masquée afin de simuler des routes inconnues. Les modèles apprennent à partir du réseau restant et doivent prédire les connexions masquées.

Les performances sont évaluées à l'aide de deux métriques principales :

- **AUC (Area Under the ROC Curve)** : mesure la capacité du modèle à distinguer les liens existants des liens inexistants.
- **AP (Average Precision)** : mesure la qualité du classement des liens prédits en tenant compte de la précision et du rappel.

## 4. Installation

### 4.1. Prérequis

- Python 3.8 ou supérieur.
- `pip` ou `conda`.
- Git.
- Jupyter Notebook.
- PyTorch et PyTorch Geometric.

### 4.2. Cloner le repository

```bash
git clone https://github.com/YassineHachguer/gnn-airline-link-prediction.git
cd gnn-airline-link-prediction
```

### 4.3. Créer un environnement virtuel

**Windows :**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS :**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.4. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si nécessaire, installez PyTorch Geometric en suivant les instructions officielles adaptées à votre version de PyTorch et à votre configuration CPU/CUDA :

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Installation standard :

```bash
pip install torch-geometric
```


## 5. Structure du projet

```text
gnn-airline-link-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── airports.graphml
├── notebooks/
│   ├── airline_link_prediction.ipynb
│   └── notebook_test.ipynb
└── src/
    ├── data.py
    ├── features.py
    ├── gnn_models.py
    ├── baselines.py
    ├── training.py
    ├── evaluation.py
    └── plots.py
```

## 6. Description des fichiers source

### `src/data.py`

Chargement et conversion des données du graphe.

- `load_graph(path)` : charge le graphe depuis un fichier GraphML.
- `graph_to_tensors(G)` : convertit le graphe NetworkX en tenseurs PyTorch.
- `make_data(x, edge_index)` : construit un objet `Data` de PyTorch Geometric.

### `src/features.py`

Construction des vecteurs de caractéristiques des aéroports.

- `build_features(info, coord=True, pop=True, ctry=True)` : construit les caractéristiques des nœuds.
- `build_coords(info)` : extrait les coordonnées géographiques.

Les caractéristiques comprennent :

- **Coordonnées géographiques (3 dimensions)** : latitude et longitude converties en coordonnées sphériques 3D.
- **Population (1 dimension)** : population transformée logarithmiquement et normalisée.
- **Pays (212 dimensions)** : encodage one-hot des pays.

### `src/gnn_models.py`

Implémentation des architectures GNN.

- `make_conv(kind, in_dim, out_dim)` : crée une couche de convolution GCN, GraphSAGE ou GAT.
- `Encoder` : encode les nœuds dans un espace latent, avec une architecture GAE ou VGAE et une ou deux couches.
- `MLPDecoder` : décode les représentations latentes afin de prédire les liens.
- `build_model()` : assemble l'encodeur et le décodeur.

### `src/baselines.py`

Implémentation des méthodes de référence.

- `adamic_adar()` : heuristique fondée sur la structure du graphe.
- `MLPEncoder` : réseau de neurones utilisant uniquement les caractéristiques des nœuds.
- `run_mlp()` : entraîne et évalue le modèle MLP.

### `src/training.py`

Entraînement et évaluation des modèles.

- `evaluate(model, d)` : calcule les métriques AUC et AP.
- `train_model()` : entraîne le modèle avec un mécanisme d'early stopping.
- `test_model()` : effectue l'évaluation finale sur le jeu de test.

### `src/evaluation.py`

Pipeline expérimental complet.

- Configuration et comparaison de neuf configurations GNN.
- `select_hyperparams()` : recherche des meilleurs hyperparamètres.
- `run_all()` : exécute les expériences de comparaison et d'ablation.
- `make_table()` : génère les tableaux de résultats.
- `suspicious_links()` : identifie les liens potentiellement suspects.

L'expérimentation comprend deux étapes principales :

1. Comparaison des modèles et sélection de la meilleure configuration.
2. Ablation du meilleur modèle pour étudier l'influence des caractéristiques et des composants de l'architecture.

### `src/plots.py`

Visualisation des résultats.

- `plot_comparison()` : graphique comparatif des modèles.
- `plot_grid()` : heatmap comparant les encodeurs et les décodeurs.
- `plot_training_curves()` : courbes de perte d'entraînement et d'AUC de validation.

## 7. Données

### 7.1. Format des données

Le fichier `data/airports.graphml` contient le graphe du réseau aérien.

| Élément | Valeur |
|---|---:|
| Nombre d'aéroports | 3 363 |
| Nombre de routes non orientées | 13 547 |
| Nombre d'arêtes dans `edge_index` | 27 094 |
| Nombre de caractéristiques par aéroport | 216 |

Chaque route non orientée est représentée par deux arêtes dirigées dans `edge_index`, une dans chaque sens.

### 7.2. Attributs des nœuds

Chaque aéroport possède notamment les attributs suivants :

- `lon` : longitude.
- `lat` : latitude.
- `population` : population de la ville.
- `country` : pays.
- `city_name` : nom de la ville.

### 7.3. Construction des caractéristiques

Chaque aéroport est représenté par un vecteur de **216 caractéristiques** :

| Caractéristique | Dimensions | Description |
|---|---:|---|
| Coordonnées géographiques | 3 | Coordonnées sphériques 3D |
| Population | 1 | Valeur logarithmique normalisée |
| Pays | 212 | Encodage one-hot |
| **Total** | **216** | **Vecteur de caractéristiques** |

## 8. Guide de reproductibilité

Les étapes suivantes permettent d'exécuter les expériences depuis le notebook principal.

### Étape 1 : Chargement des données

```python
from src.data import load_graph, graph_to_tensors, make_data
from src.features import build_features

G = load_graph("data/airports.graphml")
edge_index, info = graph_to_tensors(G)

x_full = build_features(info)
data = make_data(x_full, edge_index)

print(data)
```

Sortie attendue :

```text
Data(x=[3363, 216], edge_index=[2, 27094])
```


### Étape 2 : Séparation train / validation / test

```python
from src.evaluation import split_data

train_d, val_d, test_d = split_data(
    data,
    test_ratio=0.10,
    seed=0
)
```

Exemple de répartition attendue :

| Ensemble | Liens positifs | Liens négatifs |
|---|---:|---:|
| Train | 11 516 | Selon la stratégie d'échantillonnage |
| Validation | 677 | 677 |
| Test | 1 354 | 1 354 |

Les nombres exacts dépendent de la stratégie de séparation, de l'échantillonnage des liens négatifs et de l'implémentation utilisée.

### Étape 3 : Premier entraînement

Exemple d'entraînement d'un VGAE avec un encodeur GCN et un décodeur par produit scalaire :

```python
from src.gnn_models import build_model
from src.training import train_model, test_model

model = build_model(
    x_full.shape[1],
    hidden=64,
    out_dim=32,
    n_layers=2,
    variational=True,
    conv="gcn",
    decoder="dot"
)

out = train_model(
    model,
    train_d,
    val_d,
    epochs=200,
    variational=True,
    lr=0.01
)

auc, ap = test_model(model, test_d)

print(f"Test : AUC = {auc:.4f}, AP = {ap:.4f}")
```

Les performances doivent être mesurées sur le jeu de test. Les valeurs dépendent notamment de la graine aléatoire, des versions des bibliothèques et de la configuration expérimentale.

### Étape 4 : Sélection des hyperparamètres

```python
from src.evaluation import select_hyperparams

grid = {
    "hidden": [32, 64],
    "out": [16, 32],
    "lr": [0.01, 0.005]
}

best_hp, df_hp = select_hyperparams(
    edge_index,
    info,
    grid,
    ratio=0.10,
    seeds=(0, 1)
)

print(best_hp)
```

La sélection repose sur les performances de validation afin d'identifier les hyperparamètres les plus adaptés.

### Étape 5 : Expérience complète et ablation

```python
from src.evaluation import run_all, make_table

ratios = (0.10, 0.20)

results, meta = run_all(
    edge_index,
    info,
    ratios=ratios,
    seeds=(0, 1, 2, 3, 4),
    hp=best_hp
)

table, mean, std = make_table(
    results,
    ratios,
    meta["rows"]
)

print(table.to_markdown())
```

Cette étape compare :

- Deux baselines : Adamic-Adar et MLP.
- Neuf configurations GNN : trois encodeurs combinés à trois décodeurs.
- Deux ratios de liens masqués : 10 % et 20 %.
- Cinq graines aléatoires pour mesurer la stabilité des résultats.
- Plusieurs variantes du meilleur modèle dans l'expérience d'ablation.

### Étape 6 : Visualisations

```python
from src.plots import (
    plot_comparison,
    plot_grid,
    plot_training_curves
)

plot_grid(
    mean,
    GRID_NAMES,
    CONVS,
    DECS,
    DEC_LABEL,
    "10% de liens masqués",
    "AUC"
)

plot_comparison(
    mean,
    std,
    "10% de liens masqués",
    "AUC"
)
```

### Étape 7 : Identification des liens suspects

```python
from src.evaluation import fit_final, suspicious_links

model, splits, _ = fit_final(
    meta["best_cfg"],
    edge_index,
    info,
    hp=best_hp,
    ratio=0.10,
    seed=0
)

suspicious_links(
    model,
    splits[0],
    info,
    k=15
)
```

Cette fonctionnalité permet d'identifier des paires d'aéroports dont les connexions prédites méritent une vérification supplémentaire.

Une prédiction élevée ou faible ne prouve toutefois pas qu'une route est réellement existante ou inexistante.

## 9. Configuration des expériences

Les principaux paramètres expérimentaux sont configurables depuis le notebook.

| Paramètre | Valeurs utilisées | Signification |
|---|---|---|
| `test_ratio` | `0.10`, `0.20` | Proportion de liens masqués |
| `seeds` | `(0, 1, 2, 3, 4)` | Graines aléatoires |
| `epochs` | `200` | Nombre maximal d'époques |
| `hidden` | `[32, 64]` | Dimensions cachées de l'encodeur |
| `out` | `[16, 32]` | Dimensions de l'espace latent |
| `lr` | `[0.01, 0.005]` | Taux d'apprentissage |
| `conv` | `["gcn", "sage", "gat"]` | Types d'encodeurs |
| `decoder` | `["dot", "mlp", "mlp_dist"]` | Types de décodeurs |

## 10. Résultats expérimentaux

Les valeurs ci-dessous correspondent aux résultats de référence du projet. Elles doivent être considérées comme des résultats rapportés, et non comme des performances garanties à chaque exécution.

### 10.1. Comparaison des modèles

| Modèle | AUC — 10 % masqués | AUC — 20 % masqués |
|---|---:|---:|
| Adamic-Adar | 0.939 ± 0.002 | 0.920 ± 0.003 |
| MLP | 0.965 ± 0.002 | 0.965 ± 0.002 |
| VGAE GCN + produit scalaire | 0.986 ± 0.001 | 0.985 ± 0.001 |
| VGAE GCN + MLP | 0.990 ± 0.001 | 0.987 ± 0.001 |
| VGAE GCN + MLP + distance | 0.989 ± 0.002 | 0.987 ± 0.002 |

**Meilleur modèle rapporté : VGAE avec encodeur GCN et décodeur MLP.**

### 10.2. Résultats de l'ablation

L'ablation consiste à retirer certaines caractéristiques afin d'évaluer leur contribution aux performances.

| Configuration | AUC — 10 % masqués |
|---|---:|
| Sans population | 0.992 ± 0.002 |
| Sans coordonnées | 0.986 ± 0.002 |
| Sans pays | 0.988 ± 0.001 |
| Sans aucune caractéristique | 0.884 ± 0.002 |

Ces résultats suggèrent que :

- La suppression de la population améliore les performances dans cette expérience.
- Les coordonnées géographiques contribuent à la prédiction.
- L'information sur le pays apporte également une contribution importante.
- L'absence de caractéristiques dégrade les performances, ce qui souligne l'intérêt des informations associées aux aéroports.

Ces conclusions restent propres au jeu de données et au protocole expérimental employés.

## 11. Interprétation des résultats

### Performance globale

Les résultats rapportés montrent la hiérarchie suivante :

- GNN : AUC proche de 0.990.
- MLP : AUC proche de 0.965.
- Adamic-Adar : AUC proche de 0.939.

Le MLP exploite les caractéristiques des aéroports sans effectuer de propagation de l'information dans le graphe. Adamic-Adar utilise la structure du graphe, notamment les voisins communs, sans exploiter directement les caractéristiques des nœuds.

Le GNN combine ces deux sources d'information : les caractéristiques des aéroports et la structure du réseau.

### Importance des caractéristiques

- **Coordonnées géographiques** : elles renseignent sur la proximité spatiale entre les aéroports.
- **Pays** : elles permettent au modèle d'exploiter les relations entre les aéroports appartenant à un même pays ou à des pays différents.
- **Population** : elle n'améliore pas nécessairement la prédiction et peut introduire du bruit ou des informations peu utiles dans ce contexte.

### Choix de l'architecture

- **GCN** : meilleur encodeur dans les résultats rapportés.
- **MLPDecoder** : meilleur décodeur que le produit scalaire simple dans l'expérience principale.
- **Distance géographique** : son apport supplémentaire reste limité dans les résultats présentés.

### Robustesse

L'utilisation de plusieurs graines aléatoires permet d'étudier la variabilité des performances.

Le maintien de résultats élevés lorsque 20 % des liens sont masqués suggère que les modèles sont capables de retrouver une grande partie de la structure du réseau à partir des connexions restantes.

## 12. Dépendances

Les principales bibliothèques utilisées sont :

| Bibliothèque | Utilisation |
|---|---|
| `networkx` | Manipulation des graphes et heuristiques |
| `scikit-learn` | Métriques d'évaluation |
| `pandas` | Manipulation des résultats |
| `numpy` | Calcul numérique |
| `matplotlib` | Visualisation |
| `seaborn` | Visualisations statistiques |
| `torch` | Entraînement des réseaux de neurones |
| `torch-geometric` | Implémentation des modèles GNN |

Les versions exactes des dépendances sont définies dans `requirements.txt`.

## 13. Points clés du projet

- **Approche systématique** : comparaison de deux baselines et de neuf configurations GNN.
- **Évaluation rigoureuse** : comparaison de plusieurs ratios de liens masqués et de plusieurs graines aléatoires.
- **Ablation** : analyse de l'importance des différentes caractéristiques.
- **Reproductibilité** : paramètres expérimentaux documentés.
- **Application pratique** : identification de liens potentiellement intéressants pour une vérification ultérieure.
- **Analyse comparative** : étude de différentes architectures d'encodeurs et de décodeurs.

