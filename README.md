# Projet 7 du parcours Data Science d'Openclassrooms

Dans le cadre de ma formation avec Openclassrooms, j'ai implémenté un outil de scoring crédit pour une société financière fictive.

# Organisation 

Ce répository contient le code de cet outil. Il est séparé en :
- Le répertoire `/api` contient les routines utilisées pour répondre aux requêtes HTTP par FastAPI.
- Le répertoire `/spa` contient le dashboard sous forme de Single-Page Application
- Le répertoire `/data` contient les données sous forme brutes (input) et transformées (output) pour apprentissage de ML.
- Le répertoire `/etl` contient les routines de Extract-Transform-Load pour changer le format des données (préprocessing et de feature engineering)
- Le répertoire `/model` contient l'image du modèle ML à utiliser par l'API ainsi que du code pour construire le modèle de Machine Learning approprié.

# Installation

Pour installer, placer les fichier de données de https://www.kaggle.com/competitions/home-credit-default-risk/data dans le répertoire `/data/input`.
Puis, exécuter ces deux programmes `python -m etl.pipeline` et `python -m etl.send_to_db` pour remplir le répertoire `/data/output`.

Ensuite, exécuter `docker build --tag <your_image_tag> . ` et `docker run -d -p 80:80 <your_image_tag>`.