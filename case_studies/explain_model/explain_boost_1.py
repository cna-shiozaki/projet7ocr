import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import  metrics, model_selection
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import fbeta_score, make_scorer

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMB_Pipeline

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
from case_studies.select_model.utils import Mock
from case_studies.select_model.clean_functions import clean_up_nans

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
loans_df = pd.read_csv('../data/output/loans.csv', nrows=None )

# Nettoyage des NaN
clean_up_nans(loans_df, "merge_with_bureau", ["BURO_","ACTIVE_","CLOSED_"])
clean_up_nans(loans_df, "merge_with_prev", ["PREV_","APPROVED_","REFUSED_"])
clean_up_nans(loans_df, "merge_with_pos", ["POS_"])
clean_up_nans(loans_df, "merge_with_ins", ["INSTAL_"])
clean_up_nans(loans_df, "merge_with_cc", ["CC_"])


# Remplacer les NaN restants par 0, et se débarasser des 3 colonnes contenant inf
loans_df.fillna( value=0, inplace=True )
cols_withs_infs_series = pd.DataFrame(np.isinf(loans_df.values), columns=loans_df.columns).sum()
cols_to_drop = list(cols_withs_infs_series.loc[ cols_withs_infs_series > 0 ].index)
loans_df.drop(labels=cols_to_drop,axis="columns",inplace=True)


# Séparation données de travail et données d'examination
train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)
assert len( train_test_boundaries_list) == 2

loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]].copy()
loans_exam_df = loans_df.iloc[train_test_boundaries_list[1]:]

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


################################
####### PIPELINE ###############
################################

# 1/ Smote
smoteenn = SMOTEENN(random_state=0)
X_train, y_train = smoteenn.fit_resample(X_train, y_train)

# 2/ Standard Scaling
scaler = StandardScaler()
scaler.fit(pd.concat([X_train,X_test]))
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# 3/ GradientBoosting
clf = HistGradientBoostingClassifier()

# Let's go !
clf.fit(X_train , y_train )

y_predicted = clf.predict(X_test)
y_predicted_proba = clf.predict_proba(X_test)

auc_test = round(metrics.roc_auc_score(y_true=y_test,y_score=y_predicted_proba.T[1]),2)
print("ROC AUC Score (sur données de test) = ",auc_test)



#############################
##### EXPLAINABILITY ########
#############################
# SHAP Interaction Values

### TREE
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# 1 / Force Plot
shap.force_plot(explainer.expected_value, shap_values[1, :], X_test.iloc[1])
# 2 / Summary Plot
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")


## GRADIENT
explainer = shap.GradientExplainer(clf, X_train)
shap_values = explainer.shap_values(X_test)
# 1 / Force Plot
shap.force_plot(explainer.expected_value, shap_values[1, :], X_test.iloc[1])

# Shap Shit
X100 = shap.utils.sample(X_test, 100) # 100 instances for use as the background distribution
explainer_histgb = shap.Explainer(clf, X100)

shap_values_histgb = explainer_histgb(X_test, check_additivity=False)


