import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import  metrics, model_selection
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import fbeta_score, make_scorer

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMB_Pipeline

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
from case_studies.select_model.utils import Mock, in_ipython
from etl.clean_up import take_care_of_nan, work_exam_split


# Chargement du dataset
loans_df = pd.read_csv('data/output/loans.csv', nrows=None )

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


# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


#####################################


smote_enn = SMOTEENN(random_state=0)
X_train, y_train = smote_enn.fit_resample(X_train, y_train)


print("### DUMMY MODEL #####")
clf = make_pipeline(StandardScaler(),
                    DummyClassifier() )

clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)
y_predicted_proba = clf.predict_proba(X_test)

# Afficher la précision (accuracy)
print("Précision = ", metrics.accuracy_score(y_test, y_predicted))
print("Rappel = ", metrics.recall_score(y_test, y_predicted))

# 1/ Calculer la courbe ROC
fp_rate, tp_rate, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predicted_proba.T[1], pos_label=None)

# DataFrame de la courbe ROC
fpr_tpr_df = pd.DataFrame(index=thresholds, data={"FPR": fp_rate,"TPR": tp_rate} )
fpr_tpr_df.index.name = "seuil"

# 2/ Calculer l'aire sous la courbe
auc = metrics.roc_auc_score(y_true=y_test, y_score=y_predicted_proba.T[1])
print("Aire sous la courbe =",str(round(auc,2)))

# 3/ Affichier la courbe ROC
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fp_rate,tp_rate,label="data 1, auc="+str(auc))
plt.legend(loc=4)


# Si tu veux au moins 90% de TP / P ...
C_MIN_TP_P = 0.90
min_fpr_tpr_row = fpr_tpr_df.loc[fpr_tpr_df.TPR > C_MIN_TP_P].iloc[0]
seuil_min = list(fpr_tpr_df.loc[fpr_tpr_df.FPR ==  min_fpr_tpr_row.FPR].index)[0]
print("Si tu veux au moins TP/P > à", C_MIN_TP_P, "tu vas devoir mettre un seuil de", seuil_min,"et accepter d'avoir un taux de",round(min_fpr_tpr_row.FPR,2),"faux positifs (FP/N)" )
