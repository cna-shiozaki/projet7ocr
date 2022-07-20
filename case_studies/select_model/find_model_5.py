import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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

start_time = time.time()
print("--- Let's start the preprocessing ------")

# Chargement du dataset
loans_df = pd.read_csv('./data/output/loans.csv' )

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
C_NB_SAMPLE = 20000
random_integers_for_sample = sample_without_replacement(n_population=len(y), n_samples=C_NB_SAMPLE)
X = X.iloc[random_integers_for_sample]
y = y.iloc[random_integers_for_sample]


# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

after_preprocessing_time = time.time()

print("--- %s seconds --- : Preprocessing Done" % int(after_preprocessing_time - start_time) )
print("--- Let's start the search of the best model with GridSearchCV & Pipeline ")

################################
####### PIPELINE ###############
################################

#pipe_svc = Pipeline(steps=[("smote",smote),  ("scaler", scaler), ("svc", svc)])
pipe_svc = IMB_Pipeline( steps=[("smote",SMOTEENN(random_state=0)),  ("scaler", StandardScaler()), ("estimator", Mock())] )


# Fixer les valeurs des hyperparamètres à tester
search_space = [
    {   'smote__sampling_strategy' : [0.5],
        'estimator': [RandomForestClassifier()],
        'estimator__criterion' : ["gini","log_loss"],
        'estimator__max_depth' : [None],
        'estimator__n_estimators' : [100, 500],
        'estimator__n_estimators' : ["sqrt","log2"]
    },
    {   'smote__sampling_strategy' : [0.5],
        'estimator' : [LGBMClassifier()],
        'estimator__num_leaves' : [31, 34],
        'estimator__max_depth' : [-1, 8],
        'estimator__n_estimators' : [100, 500],
        'estimator__reg_alpha' : [0, 0.041545473],
        'estimator__reg_lambda' : [0, 0.0735294]
    }    
]


def my_profitability_score(y_true, y_pred_proba):
    best_threshold_score = -np.inf
    for threshold in np.linspace(start=0,stop=1 , num=10 ):
        y_pred = list(map( lambda el : 1 if el > threshold else 0 , y_pred_proba))

        nb_vrais_negatifs =  sum(starmap( lambda true, pred : true == pred == 0  , zip(y_true,y_pred)))
        nb_faux_negatifs =  sum(starmap( lambda true, pred : ( pred == 0 ) and ( true ==  1)  , zip(y_true,y_pred)))

        threshold_score = nb_vrais_negatifs - 10 * nb_faux_negatifs 

        best_threshold_score = max(best_threshold_score, threshold_score) 

    return ( best_threshold_score / len(y_true) ) * 1000   

def find_best_threshold(y_true, y_pred_proba):
    best_threshold = 0
    best_threshold_score = -np.inf
    for threshold in np.linspace(start=0,stop=1 , num=100 ):
        y_pred = list(map( lambda el : 1 if el > threshold else 0 , y_pred_proba))

        nb_vrais_negatifs =  sum(starmap( lambda true, pred : true == pred == 0  , zip(y_true,y_pred)))
        nb_faux_negatifs =  sum(starmap( lambda true, pred : ( pred == 0 ) and ( true ==  1)  , zip(y_true,y_pred)))

        threshold_score = nb_vrais_negatifs - 10 * nb_faux_negatifs

        if (threshold_score > best_threshold_score):
            best_threshold_score = max(best_threshold_score, threshold_score) 
            best_threshold = threshold
    return best_threshold    


# Utilisons mon un score spécifique, qui favorise le rappel (TP/P)
my_score = make_scorer(my_profitability_score, greater_is_better=True, needs_proba=True, needs_threshold=False)

# Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
grid_search = model_selection.GridSearchCV(
    pipe_svc,                                    # un classifieur Support Vector Machine
    search_space,                                # hyperparamètres à tester
    n_jobs=None if in_ipython() else -1,         # tous mes CPUs !!
    cv=5,                                        # nombre de folds de validation croisée
    scoring=my_score                             # score à maximiser
)



# Let's go !
grid_search.fit( X_train, y_train )

after_search_time = time.time()
print("--- %s seconds --- : GridSearchCV Done" % int(after_preprocessing_time - start_time))

# Meilleur estimateur, meilleur score
print("Meilleur estimateur" + str(grid_search.cv_results_['params'][grid_search.best_index_]))
print("Meilleur score (My Profitability) =",round(grid_search.best_score_,2))
_, meilleur_estimateur = grid_search.best_estimator_.steps[-1]

######################
### JEU DE TEST ######
######################
y_predicted = grid_search.predict(X_test)
y_predicted_proba =  grid_search.predict_proba(X_test)

auc_test = round(metrics.roc_auc_score(y_true=y_test,y_score=y_predicted_proba.T[1]),2)
print("ROC AUC Score (sur données de test) = ",auc_test)

def results_at_threshold( y_proba, threshold, print_details):
    """ Afficher report de classification + matrice de confusion au seuil 'threshold' """
    y_pred_at_threshold = list(map( lambda el : 1 if el > threshold else 0 , y_proba))
    conf_mat = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred_at_threshold)
    score_at_this_threshold = ( conf_mat[0][0]- 10 * conf_mat[1][0] ) * 1000 / len( y_proba ) 

    if print_details == True:
        print("Classification report au seuil ", threshold)
        print(metrics.classification_report(y_test, y_pred_at_threshold,target_names=["negatif","positif"]))

        print("Matrice de confusion au seuil ", threshold, "sur", len(y_proba), "individus")
        print(pd.DataFrame(conf_mat, index=["actual negatif","actual positif"], columns=["predicted negatif","predicted positif"] ))

        print("\nScore à ce seuil =", score_at_this_threshold )

    return score_at_this_threshold 

def plot_score_evolution( y_proba):
    scores = []
    thresholds_arr = np.linspace(0,1,100)
    for threshold in thresholds_arr:
        scores.append( results_at_threshold( y_proba, threshold, print_details=False))

    plt.figure(figsize=(10,5))
    plt.plot( np.linspace(0,1,100), scores, color="firebrick")
    plt.vlines(thresholds_arr[np.argmax(scores)],ymin=0,ymax=max(scores), color="palegreen",linestyles="dashed")  
    plt.hlines(max(scores),xmin=0, xmax=thresholds_arr[np.argmax(scores)]  , color="palegreen",linestyles="dashed")  
    plt.text(x=thresholds_arr[np.argmax(scores)],y=0,s="{:.2f}".format(thresholds_arr[np.argmax(scores)] ), color="seagreen" )
    plt.text(x=0,y=max(scores),s=str(int(max(scores))), color="seagreen" )
    plt.xlabel("Seuil")
    plt.ylabel("Score")
    plt.title("Évolution du score de profitabilité en fonction du seuil")
    #plt.show()
    plt.savefig("score_evolution.png")

plot_score_evolution(y_predicted_proba.T[1])


print("Meilleur score potentiel =", my_profitability_score(y_true=y_test, y_pred_proba=y_predicted_proba.T[1]))
print("Meilleur score obtenu au seuil =", find_best_threshold(y_test, y_predicted_proba.T[1]) )

best_threshold = find_best_threshold(y_test, y_predicted_proba.T[1])

results_at_threshold(y_predicted_proba.T[1], threshold=0.5, print_details=True)
print('\n',"#################################")
results_at_threshold(y_predicted_proba.T[1], threshold=best_threshold, print_details=True)



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


