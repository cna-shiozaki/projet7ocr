import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from itertools import starmap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import  metrics, model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import fbeta_score, recall_score, make_scorer

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMB_Pipeline
from lightgbm import LGBMClassifier

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")

from etl.clean_up import take_care_of_nan, work_exam_split
from case_studies.select_model.utils import Mock, in_ipython

# Chargement du dataset
loans_df = pd.read_csv('./data/output/loans.csv', nrows=10000 )

# Retirer les NaN
take_care_of_nan(loans_df)

# Séparation du DataFrame en work et exam
# tmp loans_work_df, loans_exam_df =  work_exam_split(loans_df)
loans_work_df = loans_df 

################################
####### FEATURES ###############
################################
# Prendre toutes les variables
all_cols_for_model = [ col for col in loans_work_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
X = loans_work_df[all_cols_for_model]
y = loans_work_df["TARGET"]


# Echantillonnage 
from sklearn.utils.random import sample_without_replacement
C_NB_SAMPLE = 1000
random_integers_for_sample = sample_without_replacement(n_population=len(y), n_samples=C_NB_SAMPLE)
X = X.iloc[random_integers_for_sample]
y = y.iloc[random_integers_for_sample]


# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_train, y_train = smote_enn.fit_resample(X_train, y_train)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Prédiction au seuil 0.5
y_pred  = clf.predict(X_test)

# Prédiction en probabilités
y_pred_proba  = clf.predict_proba(X_test)

y_true = y_test

print("Matrice de confusion au seuil 0.5")
print(pd.DataFrame(metrics.confusion_matrix(y_true=y_test,y_pred=y_pred), index=["actual negatif","actual positif"], columns=["predicted negatif","predicted positif"] ))


#X_test_resampled, y_test_resampled = smote_enn.fit_resample(X_test, y_test)

def my_profitability_score(y_true, y_pred_proba):
    best_threshold_score = -np.inf
    for threshold in np.linspace(start=0,stop=1 , num=10 ):
        y_pred = list(map( lambda el : 1 if el > threshold else 0 , y_pred_proba.T[1]))

        nb_vrais_negatifs =  sum(starmap( lambda true, pred : true == pred == 0  , zip(y_true,y_pred)))
        nb_faux_negatifs =  sum(starmap( lambda true, pred : ( pred == 0 ) and ( true ==  1)  , zip(y_true,y_pred)))

        threshold_score = nb_vrais_negatifs - 10 * nb_faux_negatifs

        best_threshold_score = max(best_threshold_score, threshold_score) 

    return best_threshold_score