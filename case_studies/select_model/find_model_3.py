import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import  metrics, model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import fbeta_score, make_scorer

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
select_kbest = SelectKBest(chi2, k=40)
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

#pipe_svc = Pipeline(steps=[("smote",smote),  ("scaler", scaler), ("svc", svc)])
pipe_svc = IMB_Pipeline( steps=[("smote",SMOTEENN(random_state=0)),  ("scaler", StandardScaler()), ("estimator", Mock())] )

# Fixer les valeurs des hyperparamètres à tester
search_space = [

    {
        'estimator': [GaussianNB()]
    },
    
    {   'estimator': [SVC()],
        'estimator__kernel' : ["rbf"]
    },
    {   'estimator': [RandomForestClassifier()],
        'estimator__max_depth' : [None],
        'estimator__n_estimators' : [100]
    },
    {
        'estimator' : [HistGradientBoostingClassifier()],
        'estimator__max_leaf_nodes' : [31, None],
        'estimator__l2_regularization' : [0, 0.1, 1],

    },
    {   'smote__sampling_strategy' : [0.4, 0.6],
        'estimator' : [LGBMClassifier()],
        'estimator__num_leaves' : [31],
        'estimator__max_depth' : [-1],
        'estimator__n_estimators' : [100],
    }
]


# Utilisons mon un score spécifique, qui favorise le rappel (TP/P)
my_fbeta_score = make_scorer(fbeta_score, beta=30)

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
grid_search = model_selection.GridSearchCV(
    pipe_svc,                                    # un classifieur Support Vector Machine
    search_space,                                # hyperparamètres à tester
    n_jobs=None if in_ipython() else -1,         # tous mes CPUs !!
    cv=5,                                        # nombre de folds de validation croisée
    #scoring=my_fbeta_score                      # score à maximiser
    scoring='roc_auc'                            # score à maximiser
)


# Let's go !
grid_search.fit( X_train, y_train )

# Meilleur estimateur, meilleur score
print("Meilleur estimateur" + str(grid_search.cv_results_['params'][grid_search.best_index_]))
print("Meilleur score (AUC) =",round(grid_search.best_score_,2))
_, meilleur_estimateur = grid_search.best_estimator_.steps[-1]

######################
### JEU DE TEST ######
######################
y_predicted = grid_search.predict(X_test)
y_predicted_proba =  grid_search.predict_proba(X_test)

auc_test = round(metrics.roc_auc_score(y_true=y_test,y_score=y_predicted_proba.T[1]),2)
print("ROC AUC Score (sur données de test) = ",auc_test)
print("Classification report au seuil 0.5")
print(metrics.classification_report(y_test, y_predicted,target_names=["negatif","positif"]))

# Afficher la matrice de confusion
# que tu peux vérifier avec : ( y_test + y_predicted == 0 ).sum() / ( y_test + y_predicted == 2 ).sum()
print("Matrice de confusion au seuil 0.5")
print(pd.DataFrame(metrics.confusion_matrix(y_true=y_test,y_pred=y_predicted), index=["actual negatif","actual positif"], columns=["predicted negatif","predicted positif"] ))

# Calcul de la courbe ROC
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predicted_proba.T[1])

# DataFrame de la courbe ROC
fpr_tpr_df = pd.DataFrame(index=thresholds, data={"FPR": fpr,"TPR": tpr} )
fpr_tpr_df.index.name = "seuil"

# 3/ Afficher la courbe ROC
plt.title(str(meilleur_estimateur))
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fpr,tpr,label="Aire sous la courbe, auc="+str(auc_test))
plt.legend(loc=4)


# Si tu veux au moins 90% de TP / P ...
C_MIN_TP_P = 0.90
min_fpr_tpr_row = fpr_tpr_df.loc[fpr_tpr_df.TPR > C_MIN_TP_P].iloc[0]
seuil_min = list(fpr_tpr_df.loc[fpr_tpr_df.FPR ==  min_fpr_tpr_row.FPR].index)[0]
print("Si tu veux au moins TP/P > à", C_MIN_TP_P, "tu vas devoir mettre un seuil de", seuil_min,"et accepter d'avoir un taux de",round(min_fpr_tpr_row.FPR,2),"faux positifs (FP/N)" )


# Courbe précision-rappel
fig, ax = plt.subplots(figsize=(15,10))
ax.set_title('Courbe Précision-Rappel')
metrics.PrecisionRecallDisplay.from_predictions(y_true=y_test, y_pred=y_predicted_proba.T[1], ax=ax)

# Faire un DF de grid_search.cv_results_
pd.DataFrame(index=grid_search.cv_results_["param_estimator"].data, 
    data={ 
        "rank_test_score": grid_search.cv_results_["rank_test_score"],
        "mean_test_score": grid_search.cv_results_["mean_test_score"],
        "std_test_score": grid_search.cv_results_["std_test_score"],
        "mean_fit_time": grid_search.cv_results_["mean_fit_time"],
        "std_fit_time": grid_search.cv_results_["std_fit_time"],
        "mean_score_time": grid_search.cv_results_["mean_score_time"],
        "std_score_time": grid_search.cv_results_["std_score_time"]
    }  )


