import json
from dataclasses import dataclass, field

from api.prediction_engine import PredictionEngine
from api.data_access import DataAccess
from api.bbox_maker import BoxPlotMaker

@dataclass
class Projet7Server():
    prediction_engine : PredictionEngine = None
    data_access : DataAccess = None
    box_plot_maker : BoxPlotMaker = None

    def initialize(self):
        """ Initialiser les trois composants :
        - une instance de DataAccess faisant office de base de données des demandes de prêts. 
        - une instance de PredictionEngine permettant de faire de la classification  
        - une instance de BoxPlotMaker permettant de faire de tracer des boxplots  """
        self.data_access = DataAccess()

        self.prediction_engine = PredictionEngine()
        self.prediction_engine.prepare_for_use(model_name="boost_model")

        self.box_plot_maker = BoxPlotMaker()

    def fetch_customer_prediction(self, loan_id):
        """ Aller chercher le score de prédiction pour un seul client (celui identifié par 'loan_id') """
                    
        loan_series = self.data_access.query_from_loan_id( loan_id=loan_id )

        # Aller lire la prédiction selon le modèle
        prediction, prediction_value = self.prediction_engine.single_predict_using_model( loan_series )
            
        # Renvoyer la prédiction en pourcentage [0-100] et pas en probabilité [0-1]
        return prediction, round(prediction_value * 100)
        

    def fetch_customer_details( self, loan_id) :
        """ Aller chercher les détails du client identifié par la demande de prêt 'loan_id' """
        loan_series = self.data_access.query_from_loan_id( loan_id=loan_id )

        return {
            "sk_id_curr": int(loan_series["SK_ID_CURR"]),
            
            "amt_credit" : loan_series["AMT_CREDIT"],
            "amt_income_total" : loan_series["AMT_INCOME_TOTAL"],
            "amt_credit" : loan_series["AMT_CREDIT"],
            "amt_annuity" : loan_series["AMT_ANNUITY"],
            "amt_goods_price" : loan_series["AMT_GOODS_PRICE"],
            "payment_rate" : loan_series["PAYMENT_RATE"],
            
            
            "code_gender" : int(loan_series["CODE_GENDER"]) ,
            "cnt_children" : int(loan_series["CNT_CHILDREN"]),
            "days_birth" : abs( int(  loan_series["DAYS_BIRTH"])),
            "name_family_status" : get_categorical_name(loan_series, "NAME_FAMILY_STATUS"),
            "name_income_type": get_categorical_name(loan_series, "NAME_INCOME_TYPE"), 
            "occupation_type": get_categorical_name(loan_series, "OCCUPATION_TYPE"),
            "organization_type": get_categorical_name(loan_series, "ORGANIZATION_TYPE"),
            "name_contract_type": get_categorical_name(loan_series, "NAME_CONTRACT_TYPE"),
            
            "flag_own_car": int(loan_series["FLAG_OWN_CAR"]),
            "flag_own_realty": int(loan_series["FLAG_OWN_REALTY"]),
            }


    def fetch_random_loan_id(self):
        return self.data_access.query_random( )

    def fetch_lime_tabular(self, loan_id):

        loan_series = self.data_access.query_from_loan_id( loan_id=loan_id )
        
        return json.dumps(self.prediction_engine.single_lime_tabular_explain(loan_series) )
        

    def fetch_box_plot(self, loan_id):
        loan_series = self.data_access.query_from_loan_id( loan_id=loan_id )

        return self.box_plot_maker.build_boxplot(loan_series)

    def fetch_specific_field(self, loan_id, specific_field):
        loan_series = self.data_access.query_from_loan_id( loan_id=loan_id )

        return loan_series.loc[specific_field]


    def fetch_column_names(self):
        return json.dumps( [ { "column_name" : e } for e in self.data_access.query_all_columns_names()  ] )


def get_categorical_name(loan_series, prefix):
    appropriate_cols = [ col for col in loan_series.index if col.startswith(prefix)]
    my_series = (loan_series.loc[appropriate_cols] == 1  )

    actual_valued_col = my_series.loc[ my_series == True ]
    if len(actual_valued_col) == 0:
        return 'Unknown'
    elif len(actual_valued_col) == 1:
        return actual_valued_col.index[0].replace(prefix+'_','')
    else:
        return 'Several'
    
