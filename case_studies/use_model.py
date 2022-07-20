import os
import pickle
import pandas as pd

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")

from etl.clean_up import take_care_of_nan, work_exam_split

# Chargement du dataset
loans_df = pd.read_csv('./data/output/loans.csv', nrows=500 )

# Juste le train dataset..
take_care_of_nan(loans_df)

filepath_of_the_model = "./model/dumps/boost_model.sav"

# Chargement dans l'attribut 'model' (TODO : utiliser 'with')
file_of_the_model = open(filepath_of_the_model,"rb")
model = pickle.load(file_of_the_model)
file_of_the_model.close()

clf_name, clf = model.steps[-1]

df = pd.DataFrame({"name": model.feature_names_in_, "importance": clf.feature_importances_})
df.sort_values(by="importance", ascending=False).head(10)

# NOPE !
#all_cols_for_model = [ col for col in loans_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
#X = loans_df[all_cols_for_model]

# YES !
X = loans_df[model.feature_names_in_]

import dill
from lime import lime_tabular

filepath_of_the_lime_explainer = "./model/dumps/lime_tabular_explainer.sav"

# Lecture du Explainer par dans le fichier model.sav
output_filename = "model/dumps/lime_tabular_explainer.sav"

with open(output_filename, 'rb') as f:
   lime_tabular_explainer = dill.load(f)


lime_exp = lime_tabular_explainer.explain_instance(
    data_row=X.iloc[0],
    predict_fn=model.predict_proba
)
lime_exp.show_in_notebook(show_table=True)



# Oublie oublie
cols_to_keep = ['CODE_GENDER',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'EXT_SOURCE_1',
 'EXT_SOURCE_2',
 'EXT_SOURCE_3',
 'NAME_INCOME_TYPE_Pensioner',
 'NAME_INCOME_TYPE_Working',
 'NAME_EDUCATION_TYPE_Higher education',
 'OCCUPATION_TYPE_Laborers',
 'OCCUPATION_TYPE_nan',
 'ORGANIZATION_TYPE_XNA',
 'BURO_DAYS_CREDIT_MIN',
 'BURO_CREDIT_ACTIVE_Closed_MEAN',
 'CLOSED_DAYS_CREDIT_MIN',
 'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
 'PREV_CODE_REJECT_REASON_SCOFR_MEAN',
 'PREV_NAME_PRODUCT_TYPE_walk-in_MEAN',
 'REFUSED_HOUR_APPR_PROCESS_START_MAX',
 'REFUSED_HOUR_APPR_PROCESS_START_MEAN']