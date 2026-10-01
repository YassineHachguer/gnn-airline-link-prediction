# Exploration des données — Graphe des aéroports

Ce document présente les résultats de l'exploration du graphe des aéroports, première étape du projet décrit dans le [README](README.md). Il synthétise ce que les données révèlent, ce qu'il faut nettoyer et quelles caractéristiques construire avant toute modélisation.

L'analyse complète, avec le code et les figures, se trouve dans `notebooks/01_exploration.ipynb`.

---

## Démarche

L'exploration a été conduite comme une étude scientifique et non comme un simple calcul de statistiques. Elle est organisée en sept questions (Q0 à Q6). Pour chacune :

1. une hypothèse est formulée **avant** l'exécution ;
2. le calcul est effectué ;
3. le résultat est interprété et traduit en **décision** de prétraitement, de modélisation ou d'évaluation.

```text
   Question + hypothèse
            │
            ↓
         Calcul
            │
            ↓
   Observation (confirmée / réfutée)
            │
            ↓
   Décision (cleaning, features, protocole)
```

Aucune donnée n'est modifiée lors de l'exploration : les problèmes sont uniquement recensés. Les transformations seront appliquées par un script dédié (`src/clean.py`), afin que chaque modification soit traçable et justifiée.

---

## Description du jeu de données

Le fichier `airportsAndCoordAndPop.graphml.xml` décrit un graphe **non orienté** :

| Élément | Nombre | Description |
|---|---|---|
| Nœuds | 3 363 | Aéroports |
| Arêtes | 13 547 | Routes aériennes |
| Pays | 211 | Après normalisation des noms |

Chaque aéroport possède cinq attributs : `lat`, `lon`, `population`, `country`, `city_name`.

Le fichier ne contient **aucun attribut sur les routes** (ni distance, ni trafic, ni compagnie) et **aucune dimension temporelle**. La distance d'une route doit donc être calculée à partir des coordonnées, et la prédiction de liens porte sur un graphe statique : on retrouve des routes masquées, on ne prédit pas des routes futures.

---

## Résultats

### 1. Le fichier est structurellement propre (Q0)

* aucune boucle ni arête dupliquée ;
* aucune valeur manquante au sens strict ;
* les identifiants des nœuds ne sont pas contigus (10 numéros absents) ;
* 9 noms de pays sont mal formés (`M&Eacute;XICO#MEXICO`, `VI&Ecirc;T_NAM#VIET_NAM,VIETNAM`, etc.).

### 2. Le réseau est fortement structuré localement (Q1)

* le coefficient de clustering moyen vaut **0,49**, soit environ **220 fois** celui d'un graphe aléatoire de même taille : deux aéroports ayant des voisins communs sont très souvent reliés ;
* l'assortativité de degré est quasi nulle (0,04). **L'hypothèse d'une structure en étoile (hub-and-spoke) est réfutée** : le réseau combine vraisemblablement un cœur de hubs interconnectés et des réseaux régionaux.

### 3. Les degrés sont très inégaux (Q2)

* degré médian de 3, degré maximal de 248 (Paris, Londres, Francfort, Amsterdam, Chicago) ;
* **21,6 %** des aéroports n'ont qu'une seule route ;
* la composante connexe principale regroupe **99,6 %** des nœuds.

![Distribution des degrés](figures/fig_degree_distribution.pdf)

La distribution est à queue lourde avec une coupure pour les degrés élevés ; on ne parlera donc pas de loi de puissance.

### 4. La distance est le facteur le plus déterminant (Q3)

| | Routes existantes | Paires aléatoires non reliées |
|---|---|---|
| Distance médiane | **814 km** | **8 958 km** |
| Part à moins de 1 000 km | 55,7 % | 2,0 % |

Les deux distributions sont presque disjointes (test de Kolmogorov-Smirnov : D = 0,71).

![Distance des routes vs paires aléatoires](figures/fig_distance_edges_vs_random.pdf)

**Conséquence majeure :** si l'on évalue les modèles avec des paires aléatoires comme exemples négatifs, la distance suffit presque à elle seule à distinguer les vraies routes. Le protocole d'évaluation doit en tenir compte.

### 5. Les routes restent majoritairement à l'intérieur d'un pays (Q4)

* **57,4 %** des routes sont domestiques, contre 5,4 % attendus au hasard (facteur 10,7) ;
* 62 pays ne comptent qu'un seul aéroport.

### 6. L'attribut population est peu fiable (Q5)

* **49,7 %** des aéroports ont une population de 10 000 exactement, qui est une valeur par défaut ;
* 61 noms de ville partagés par plusieurs aéroports ont exactement la même population (par exemple Athens en Grèce et aux États-Unis), ce qui indique un appariement réalisé sur le seul nom de la ville ;
* la corrélation entre population et degré est modérée (Spearman = 0,39) ;
* les aéroports à population par défaut ont un degré moyen de **4,5**, contre **11,6** pour les autres : l'absence de population est elle-même informative.

### 7. Certains aéroports sont mal localisés (Q6)

Des aéroports possèdent des coordonnées et un pays erronés, alors que **leurs routes sont correctes**. L'examen de leurs voisins permet de le vérifier :

| Aéroport | Localisation dans le fichier | Voisins | Localisation réelle |
|---|---|---|---|
| Hasvik | Nouvelle-Zélande | Tromsø, Hammerfest | Norvège |
| Hagfors | Océan Pacifique | Stockholm | Suède |
| Pietersburg | États-Unis | Johannesburg | Afrique du Sud |
| Camaguey | Australie | La Havane, Rome, Buenos Aires | Cuba |
| Burao | États-Unis | Djibouti, Berbera | Somalie |

Ces erreurs touchent également des aéroports importants (Ottawa, 26 routes ; Jersey, 18 routes), ce qui fausse la distance de nombreuses arêtes.

Un **score de suspicion** a été construit pour les détecter : il compare la distance d'un aéroport à ses voisins avec la distance de ces voisins à leurs propres voisins. Les 30 aéroports les plus suspects sont listés dans `outputs/suspect_nodes.csv`.

---

## Ce que l'exploration implique pour le projet

1. **Le protocole d'évaluation doit utiliser des négatifs difficiles** (paires proches géographiquement mais non reliées). Avec des négatifs aléatoires, la tâche se réduit presque à un test de distance et la comparaison des modèles perd son sens.
2. **Les baselines simples seront fortes** : Common Neighbors et Adamic-Adar (grâce au clustering élevé), Preferential Attachment (grâce aux hubs) et une baseline **distance seule**, ajoutée à la suite de cette exploration. Le GNN devra les dépasser pour être jugé pertinent.
3. **La population doit être utilisée avec prudence**, accompagnée d'un indicateur de valeur manquante.
4. **Les aéroports mal localisés constituent une vérité terrain** exploitable pour un volet de détection d'anomalies.

---

## Nettoyage à réaliser (`src/clean.py`)

| Action | Justification |
|---|---|
| Réindexer les nœuds de 0 à N−1 et conserver une table de correspondance | Exigence de PyTorch Geometric ; identifiants non contigus |
| Normaliser les noms de pays | 9 chaînes mal formées |
| Conserver uniquement la composante connexe principale | 12 nœuds répartis en 5 petites composantes |
| **Signaler** (sans corriger) les aéroports mal localisés | Préserver la vérité terrain du volet anomalies |
| **Signaler** (sans corriger) les populations par défaut | L'absence de population est informative |

Les arêtes ne sont pas modifiées : elles sont toutes correctes. Les coordonnées et les populations ne sont pas corrigées manuellement, car une telle correction serait coûteuse, difficile à justifier et effacerait les anomalies.

---

## Caractéristiques à construire

### Caractéristiques des aéroports (matrice `x`)

| Caractéristique | Source | Justification |
|---|---|---|
| sinus et cosinus de la latitude et de la longitude | `lat`, `lon` | La longitude est cyclique (−180° et 180° sont voisins) |
| log(population) | `population` | Valeurs comprises entre 600 et 22 millions |
| indicateur « population manquante » | `population = 10 000` | Associé à un degré nettement plus faible |
| degré | graphe d'entraînement | Les hubs attirent les routes |
| indicateur « aéroport suspect » | score de suspicion | Optionnel, à évaluer par ablation |

### Caractéristiques des paires d'aéroports (décodeur)

| Caractéristique | Justification |
|---|---|
| distance haversine | Facteur le plus discriminant observé |
| « même pays » (0/1) | Routes domestiques dix fois plus fréquentes qu'au hasard |

> **Point de vigilance — fuite de données.** Le degré doit être calculé **uniquement sur le graphe d'entraînement**, après la séparation des liens. Sinon, les routes masquées du jeu de test seraient indirectement visibles par le modèle et les performances seraient surestimées.

Chaque caractéristique sera retirée à tour de rôle lors de l'étude d'ablation afin de vérifier son utilité réelle.

---

## Décisions à prendre en équipe

Avant l'écriture de `clean.py`, deux points restent à trancher :

1. **Traitement des aéroports mal localisés dans le jeu de test** : exclure les routes concernées de l'évaluation, ou mesurer explicitement leur effet sur les performances.
2. **Définition des négatifs difficiles** : seuil de distance (par exemple, paires à moins de 1 000 km non reliées) ou plus proches voisins géographiques de chaque aéroport.

---

## Prochaines étapes

```text
Exploration (terminée)
        ↓
clean.py ── données nettoyées (data/clean/)
        ↓
split.py ── séparation des liens connus / cachés (data/splits/)
        ↓
features.py ── construction des caractéristiques
        ↓
Baselines puis modèles GNN
```
