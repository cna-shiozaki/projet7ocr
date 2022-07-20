import os
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics 

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
loans_df = pd.read_csv('../data/output/loans.csv', nrows=None )

# Séparation données de test et données d'entrainement
train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)

assert len( train_test_boundaries_list) == 2

loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]].copy()
loans_exam_df = loans_df.iloc[train_test_boundaries_list[1]:]

# Suppression des lignes qui contiennent des Nan
cols_to_check_nan = ["AMT_ANNUITY", "AMT_GOODS_PRICE"]
loans_work_df.dropna( subset=cols_to_check_nan, inplace=True )

# Séparation jeu d'entrainement / de test 
loans_train_df, loans_test_df =  train_test_split(loans_work_df, train_size=0.8)

# Identification des colonnes numériques à garder pour le modèle
num_cols_to_keep = [ 
    "CODE_GENDER",  
    "FLAG_OWN_CAR"  , 
    "FLAG_OWN_REALTY"  , 
    "CNT_CHILDREN"  , 
    "AMT_ANNUITY" ,
    "AMT_CREDIT" ,
    "AMT_GOODS_PRICE" , 
    "AMT_INCOME_TOTAL"  ]

train_df = loans_train_df[["TARGET"] + num_cols_to_keep].copy()

train_df_sample = train_df.sample(20000)

X = train_df_sample[num_cols_to_keep].values
y = train_df_sample["TARGET"].values

########################
### CLASS RE-BALANCE ###
########################
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

########################
### SVM CLASSIFIER #####
########################
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(),
                    SVC(random_state=0, max_iter=5000, kernel='linear',  probability=True  ))
clf.fit( X_resampled  , y_resampled)

y_pred = clf.predict(loans_test_df[num_cols_to_keep])
y_probas = clf.predict_proba(loans_test_df[num_cols_to_keep])

# Afficher la précision
print("Accuracy = ", metrics.accuracy_score(loans_test_df["TARGET"], y_pred))

# 1/ Calculer la courbe ROC
fp_rate, tp_rate, thresholds = metrics.roc_curve(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1], pos_label=None)

# 2/ Calculer l'aire sous la courbe
auc = metrics.roc_auc_score(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1])

# 3/ Affichier la courbe ROC
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fp_rate,tp_rate,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()