import sqlite3
import pandas as pd

from dataclasses import dataclass, field

@dataclass
class DataAccess():

    db_path : str = "data/output/loan_projet7.db"

    def query_from_loan_id(self, loan_id : int) -> pd.Series:
        # Se connecter à la base de données
        conn = sqlite3.connect(self.db_path)

        # Attention aux SQL-injection 
        if ( len( str(loan_id) ) != 6 )  or ( not isinstance(loan_id, int) ):
            raise Exception 

        # Préparer la requête 
        query = ('SELECT * FROM loans '
                ' WHERE SK_ID_CURR = ' ) + str(loan_id)
        
        
        # Exécuter une requête (et la récupérer dans un DF)
        result_df = pd.read_sql_query(query, conn )        
        
        if len(result_df) != 1:
            raise Exception("Ce numéro de demande de prêt n'a pas été trouvé dans la base de données")

        # Fermer la connection
        conn.close()

        return result_df.iloc[0]


    def load_data_old(self):
        # Chargement du dataset
        #os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")
        loans_df = pd.read_csv('data/output/loans.csv' )

        """ Charger les données dans l'objet """
        # Séparation données de test et données d'entrainement
        train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)
        assert len( train_test_boundaries_list) == 2

        loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]]
        self.exam_data = loans_df.iloc[train_test_boundaries_list[1]:]

        #self.training_data, self.testing_data =  train_test_split(loans_work_df, train_size=0.8)

    def query_old(self, filters):
        df = self.exam_data
        return df.loc[ df.SK_ID_CURR == int( filters["loan_id"] )]

    def query_random(self):
        
        # Se connecter à la base de données
        conn = sqlite3.connect(self.db_path)

        # Exécuter une requête (et la récupérer dans un DF)
        result_df = pd.read_sql_query("SELECT SK_ID_CURR FROM loans ORDER BY RANDOM() LIMIT 1", conn )

        # Fermer
        conn.close()

        if len(result_df) > 0:
            return result_df["SK_ID_CURR"].iloc[0]
        else:
            return "" 

        
