import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#from model.baseline import PredictionEngine
#prediction_engine = PredictionEngine()
#prediction_engine.prepare_for_use()

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
loans_df = pd.read_csv('../data/output/loans.csv', nrows=None )

# Séparation données de test et données d'entrainement
train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)

assert len( train_test_boundaries_list) == 2

loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]]
loans_exam_df = loans_df.iloc[train_test_boundaries_list[1]:]

loans_train_df, loans_test_df =  train_test_split(loans_work_df, train_size=0.8)

# Identification des colonnes à garder pour le modèle
cols_to_keep = [ "DAYS_BIRTH",  "PREV_NAME_CONTRACT_STATUS_Refused_MEAN"  , "BURO_DAYS_CREDIT_MEAN"  ]

# Suppression des NA
loans_train_df_without_na = loans_train_df[cols_to_keep + ["TARGET"]].dropna(axis="index",how="any")

train_sample_df = loans_train_df_without_na.sample(1000)

X = train_sample_df[cols_to_keep].values
y = train_sample_df["TARGET"].values

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, max_iter=5000))
clf.fit(X_resampled, y_resampled)


loans_test_df_without_na = loans_test_df[cols_to_keep + ["TARGET"]].dropna(axis="index",how="any").sample(1000)

X_test = loans_test_df_without_na[cols_to_keep].values
y_test = loans_test_df_without_na["TARGET"].values
 
y_pred = pd.Series(clf.predict(X_test), name="Predicted")
y_test = pd.Series(y_test, name="Actual")

pd.crosstab(y_pred, y_test )





