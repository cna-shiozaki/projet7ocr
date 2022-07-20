import os
import sqlite3

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\data\output")

# Se connecter à la base de données
conn = sqlite3.connect('loan_projet7.db')

# Créer un curseur
cur = conn.cursor()

# Exécuter une requête
query = ('SELECT SK_ID_CURR, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY'
            ' FROM loans '
            ' WHERE SK_ID_CURR = 100001' )

cur.execute(query)

# Lire le résultat
try :
    result = next(cur)
except StopIteration:
    pass