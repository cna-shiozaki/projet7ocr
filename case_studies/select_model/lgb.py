import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from imblearn.combine import SMOTEENN

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
from etl.clean_up import take_care_of_nan, work_exam_split
from case_studies.select_model.utils import Mock, in_ipython


# Chargement du dataset
loans_df = pd.read_csv('./data/output/loans.csv', nrows=None )

# Retirer les NaN
take_care_of_nan(loans_df)

# Séparation du DataFrame en work et exam
loans_work_df, loans_exam_df =  work_exam_split(loans_df)


################################
####### FEATURES ###############
################################
# Récupération des variables les plus importantes (j'ai un petit CPU)
all_cols_for_model = [ col for col in loans_work_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
frequ_transformed_loans_work_df = pd.DataFrame( MinMaxScaler().fit_transform(loans_work_df[all_cols_for_model]) , columns=all_cols_for_model ) 
select_kbest = SelectKBest(chi2, k=20)
select_kbest.fit_transform(frequ_transformed_loans_work_df, loans_work_df["TARGET"])
cols_to_keep = select_kbest.get_feature_names_out().tolist()

# Séparation variables indépendantes et variable cible
all_cols_for_model = [ col for col in loans_work_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
X = loans_work_df[cols_to_keep]
y = loans_work_df["TARGET"]


# Echantillonnage 
from sklearn.utils.random import sample_without_replacement
C_NB_SAMPLE = 10000
random_integers_for_sample = sample_without_replacement(n_population=len(y), n_samples=C_NB_SAMPLE)
X = X.iloc[random_integers_for_sample]
y = y.iloc[random_integers_for_sample]


import re
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# 1/ Smote
smoteenn = SMOTEENN(random_state=0)
X_train, y_train = smoteenn.fit_resample(X_train, y_train)

scaler = StandardScaler()
scaler.fit(X_train) 
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test) 

X_train = pd.DataFrame(data=X_train, columns=X.columns)
X_test = pd.DataFrame(data=X_test, columns=X.columns)



from lightgbm import LGBMClassifier
clf = LGBMClassifier(
    nthread=4,
    n_estimators=10000,
    learning_rate=0.02,
    num_leaves=34,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.041545473,
    reg_lambda=0.0735294,
    min_split_gain=0.0222415,
    min_child_weight=39.3259775,
    silent=-1,
    verbose=-1, )

clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 
    eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

y_predicted_proba = clf.predict_proba(X_test)



import numpy as np
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['crédit remboursé', 'défaut de paiement'],
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[0],
    predict_fn=clf.predict_proba
)

lime_exp.as_list()
lime_exp.show_in_notebook()


import dill

# Ecriture (dump) du modèle par dans le fichier model.sav
output_filename = "model/dumps/lime_tabular_explainer.sav"
with open(output_filename, 'wb') as f:
    dill.dump(lime_explainer, f)

with open('data', 'rb') as f:
   dill.load(f)