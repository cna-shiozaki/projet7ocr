import os
import sqlite3
import pandas as pd

from etl.clean_up import take_care_of_nan, work_exam_split

####################################################################
##### Ecrire le fichier .csv dans la base de données ###############
# Cela permet ensuite de n'avoir qu'à ouvrir une connexion à la DB #
# pour lire des informations relatives à un emprunt ################
####################################################################

# (En effet, en vue d'un déploiement sur serveur web conteneurisé, il faut économiser le plus de mémoire possible
# On préfère, à chaque requête HTTP entrante, à n'avoir qu'à ouvrir une base de donnée, lire un truc, et fermer la connexion,
# plutôt que de maintenir en mémoire un coûteux fichier de 2 GB)

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\data\output")

# Ouverture du DataFrame 
loans_df = pd.read_csv("loans.csv")

# Retirer les NaN
take_care_of_nan(loans_df)

# Séparation du DataFrame en work et exam
loans_work_df, loans_exam_df =  work_exam_split(loans_df)

# ON NE GARDE QUE loans_exam_df DANS LA BASE DE DONNEES. 
# Pourquoi ? Car cette DB sera utilisé pour répondre à des requêtes sur les demandes de prêts à venir 
# (pour lesquelles il n'y a pas encore d'info sur le remboursement/banqueroute du client)

# Connexion à SQLite3
conn = sqlite3.connect("loan_projet7.db")

# Dump dans la table "loans" de la base de données (tout dégager de ce qu'il y avait dedans avant)
loans_exam_df.to_sql("loans", conn, if_exists='replace', index=False)

# Bye-bye
conn.commit()
conn.close()
