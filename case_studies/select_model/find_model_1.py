import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn import neighbors, metrics, model_selection
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as IMB_Pipeline

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


# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


################################
####### PIPELINE ###############
################################

smoteenn = SMOTEENN(random_state=0)
#smote = SMOTE()
scaler = StandardScaler()
svc = SVC( probability=False )
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="prior")

#pipe_svc = Pipeline(steps=[("smote",smote),  ("scaler", scaler), ("svc", svc)])
pipe_svc = IMB_Pipeline( steps=[("smote",smoteenn),  ("scaler", scaler), ("svc", svc)] )


# Fixer les valeurs des hyperparamètres à tester
param_grid = {'svc__kernel':["poly","rbf"]}

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
grid_search = model_selection.GridSearchCV(
    pipe_svc,               # un classifieur Support Vector Machine
    param_grid,             # hyperparamètres à tester
    cv=5,                   # nombre de folds de validation croisée
    #scoring='accuracy'      # score à optimiser - ici l'accuracy (proportion de prédictions correctes)
    scoring='roc_auc'
)


# Let's go !
grid_search.fit( X_train, y_train )

# Meilleur estimateur, meilleur score
print("Meilleur estimateur" + str(grid_search.cv_results_['params'][grid_search.best_index_]))
grid_search.best_score_

y_predicted = grid_search.predict(X_test)
print("ROC AUC Score = ",round(metrics.roc_auc_score(y_true=y_test,y_score=y_predicted),2))
print(metrics.classification_report(y_test, y_predicted,target_names=["negatif","positif"]))

# Afficher la matrice de confusion
# que tu peux vérifier avec : ( y_test + y_predicted == 0 ).sum() / ( y_test + y_predicted == 2 ).sum()
pd.DataFrame(metrics.confusion_matrix(y_true=y_test,y_pred=y_predicted), index=["actual negatif","actual positif"], columns=["predicted negatif","predicted positif"] )


# Remplacer le dernier estimateur par le même estimateur, mais avec probability = True
pipe_steps = grid_search.best_estimator_.steps
best_est_name , best_est= pipe_steps[len(pipe_steps) -  1]

best_parameters_in_pipeline = [ k for k,v in grid_search.best_params_.items() if k.startswith('svc__')]
best_parameters = list(map( lambda s: s[5:], best_parameters_in_pipeline))

params_dict = {}
for best_param_in_pipeline, best_param in zip(best_parameters_in_pipeline, best_parameters):
    params_dict.update( { best_param : grid_search.best_params_["svc__kernel"] } ) 
params_dict.update(probability=True)

# Nouvel estimateur : le même, mais avec probability=True
new_best_est = type(best_est)(**params_dict)

# Retirer le dernier estimateur
pipe_steps.pop(len(pipe_steps) - 1)

# Rajouter le nouveau estimateur
pipe_steps.append( (best_est_name, new_best_est ))

# Créer une nouvelle pipeline (non nécessaire : grid_search.best_estimator_.predict_proba() marche déjà)
pipe_svc_with_proba = IMB_Pipeline( steps= pipe_steps )
pipe_svc_with_proba.fit(X_train,y_train)
y_predicted_proba = pipe_svc_with_proba.predict_proba(X_test)

# Calcul de la courbe ROC
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predicted_proba.T[1])

# DataFrame de la courbe ROC
fpr_tpr_df = pd.DataFrame(index=thresholds, data={"FPR": fpr,"TPR": tpr} )
fpr_tpr_df.index.name = "seuil"