import os
import pickle, dill
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from model.baseline import BaselineModel


@dataclass
class PredictionEngine():
    model : object = None

    def prepare_for_use(self, model_name="baseline_model"):
        """ Charger le modèle déjà pré-entrainé depuis un fichier se trouvant dans model/dumps.
        Le chargement est effectué à partir d'une image pickleisée, et le modèle est stocké dans l'attribut 'model'
        Attention ce fichier doit contenir l'import Python from model.<xxx> import <nom_du_modele>, sinon ça fait une syntax error """
        
        filepath_of_the_model = "./model/dumps/" + model_name + ".sav"

        # Chargement du Modèle dans l'attribut 'model' 
        with open(filepath_of_the_model, 'rb') as file_of_the_model:
            self.model = pickle.load(file_of_the_model)


        # Chargement du LimeExplainer dans l'attribut 'lime_tabular_explainer'
        output_filename = "model/dumps/lime_tabular_explainer.sav"
        with open(output_filename, 'rb') as f:
            self.lime_tabular_explainer = dill.load(f)

        

    def single_predict_using_model(self, loan_series : pd.Series):
        """ Prédiction du score de remboursement de crédit
        Le DataSeries 'loan_series' correspond aux détails d'une demande de prêt.
        Selon le modèle de ML utilisé, il faut adapter les variables à passer à model.predict() !
        Souvent, on prendra un subset des variables d'entrées adéquate pour le modèle.  """
                
        # Ne garder que les colonnes "utiles" pour la prédiction - i.e. celles attendues par le modèle
        loan_series = loan_series.loc[ self.model.feature_names_in_ ]
        
        # Faire un DataFrame avec les données du client (même structure que celle attendue par le modèle)
        df = pd.DataFrame(columns=loan_series.index, data=[loan_series.values] )

        C_SEUIL = 0.15

        proba = self.model.predict_proba( df )
        
        result_tuple = ( 1 if proba[0][1] > C_SEUIL else 0 , proba[0][0] ) 
        
        # Retourner la décision + la probabilité de rembourser le prêt
        return result_tuple

    def mass_predict_using_model(self, customers_df):
        """ Prédiction en masse des scores de remboursement de crédit """
        pass

    def single_lime_tabular_explain(self, loan_series : pd.Series):
        """ Obtenir des explications sur les variables qui ont influencé la prédiction par le modèle """
        
        X_row = loan_series.loc[ self.model.feature_names_in_ ]

        lime_exp = self.lime_tabular_explainer.explain_instance(
            data_row=X_row,
            predict_fn=self.model.predict_proba,
            num_samples=500  # OMG, si je ne fais pas ça, j'explose les performances = La routine prend +10 secondes à s'exécuter
        )
        return lime_exp.as_list()     