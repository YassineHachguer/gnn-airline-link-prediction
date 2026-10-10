# GNN Airline Link Prediction

## Contexte

Le transport aérien constitue un réseau complexe dans lequel les aéroports sont interconnectés par des routes aériennes. Ce réseau peut naturellement être représenté sous forme de graphe :

* les **nœuds** représentent les aéroports ;
* les **arêtes** représentent les routes aériennes reliant les aéroports ;
* les caractéristiques des aéroports, telles que leur localisation, leur pays ou leur population, apportent des informations complémentaires sur la structure du réseau.

Cette représentation sous forme de graphe permet d'étudier non seulement les caractéristiques individuelles des aéroports, mais également leurs relations et leur position dans le réseau.

---

## Problématique

Les données décrivant un réseau aérien peuvent être incomplètes : certaines routes existantes peuvent ne pas être présentes dans les données disponibles, tandis que certaines connexions peuvent être difficiles à identifier uniquement à partir des caractéristiques individuelles des aéroports.

La problématique de ce projet est donc :

> **Comment exploiter simultanément les caractéristiques des aéroports et la structure du réseau aérien afin de prédire l'existence de routes entre des paires d'aéroports qui ne sont pas directement observées ?**

Cette problématique correspond à une tâche de **Link Prediction** sur un graphe.

L'objectif n'est donc pas simplement de classifier des aéroports individuellement, mais de déterminer si une **relation entre deux nœuds** est susceptible d'exister.

---

## Pourquoi choisir ce sujet ?

La prédiction de liens constitue un problème important dans l'analyse des graphes et possède de nombreuses applications dans des réseaux réels.

Le réseau aérien constitue un cas d'étude particulièrement intéressant car l'existence d'une route peut dépendre de plusieurs facteurs à la fois :

* les caractéristiques des deux aéroports ;
* leur localisation géographique ;
* la population des villes ;
* leur pays ;
* leur niveau de connectivité ;
* les relations qu'ils entretiennent avec les autres aéroports du réseau.

Une approche traditionnelle basée uniquement sur les caractéristiques individuelles risque donc de ne pas exploiter pleinement l'information contenue dans la structure du graphe.

Ce projet permet ainsi d'étudier comment les **Graph Neural Networks (GNN)** peuvent exploiter simultanément les informations propres aux nœuds et les relations entre ces nœuds.

---

## Valeur ajoutée de la solution

La valeur ajoutée de l'approche proposée repose principalement sur l'utilisation d'une méthode capable de prendre en compte la **structure relationnelle du réseau**.

Contrairement à une approche classique qui considère chaque aéroport indépendamment, un GNN permet de construire une représentation d'un aéroport en tenant compte de ses voisins et de son environnement dans le graphe.

Le modèle peut ainsi apprendre des représentations (*embeddings*) qui combinent :

* les caractéristiques propres aux aéroports ;
* les informations provenant de leurs voisins ;
* la structure des connexions du réseau.

Ces représentations peuvent ensuite être utilisées pour estimer la probabilité qu'une route existe entre deux aéroports.

L'approche permet également de comparer l'apprentissage par GNN avec des méthodes plus simples basées sur des heuristiques de graphe. Cette comparaison permet d'étudier si l'utilisation d'un modèle appris apporte une information supplémentaire par rapport aux propriétés structurelles élémentaires du réseau.

---

## Approche proposée

Le projet suit une approche de prédiction de liens :

```text
                  Données des aéroports
                           │
                           ↓
                    Construction
                     du graphe
                           │
                           ↓
                  Prétraitement
                  des informations
                           │
                           ↓
                 Séparation des liens
                  connus / cachés
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
       Méthodes heuristiques          GNN
              │                         │
              └────────────┬────────────┘
                           ↓
                    Prédiction des
                         liens
                           │
                           ↓
                      Évaluation
```

Une partie des connexions connues est masquée afin de simuler des routes qui seraient inconnues. Le modèle apprend alors à partir du reste du réseau et doit identifier les connexions cachées.

Les performances du GNN sont ensuite comparées à celles de méthodes de référence basées sur la structure du graphe.

---

## Intérêt de l'approche GNN

Un réseau aérien n'est pas simplement un ensemble d'aéroports indépendants. Chaque aéroport est intégré dans un réseau de connexions.

Par exemple, deux aéroports peuvent avoir des caractéristiques similaires mais occuper des positions très différentes dans le réseau. À l'inverse, deux aéroports éloignés géographiquement peuvent être fortement connectés en raison de leur rôle dans le réseau.

Le GNN permet de prendre en compte cette dimension relationnelle grâce au mécanisme de **message passing** :

```text
Caractéristiques du nœud
          +
Informations des voisins
          ↓
     Message Passing
          ↓
 Représentation du nœud
          ↓
     Prédiction du lien
```

Cette capacité à intégrer simultanément les attributs des nœuds et la structure du graphe constitue le principal intérêt de l'approche proposée.

---

## Objectif du projet

L'objectif final est de construire une solution complète permettant de :

* représenter le réseau aérien sous forme de graphe ;
* préparer et nettoyer les données ;
* mettre en place un protocole fiable de Link Prediction ;
* comparer des méthodes heuristiques et des modèles GNN ;
* évaluer la capacité des modèles à retrouver des connexions cachées ;
* analyser l'influence des différentes informations utilisées par le modèle.

Le projet vise ainsi à montrer dans quelle mesure les **Graph Neural Networks peuvent être utilisés pour exploiter la structure d'un réseau aérien et effectuer de la prédiction de liens**.
