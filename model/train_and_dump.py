import pickle , dill
import pandas as pd
import numpy as np
from lime import lime_tabular

from sklearn.preprocessing import StandardScaler,  MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils.random import sample_without_replacement

from lightgbm import LGBMClassifier

from imblearn.combine import SMOTEENN


from model.baseline import BaselineModel
from etl.clean_up import take_care_of_nan, work_exam_split

# Fichier exécutable à exécuter avec " python -m model.train_and_dump " depuis la racine du projet 

print("Chargement du dataset : BEGIN")
loans_df = pd.read_csv('data/output/loans.csv' )
print("Chargement du dataset : DONE")


# Retirer les NaN
print("Nettoyage des NANs : BEGIN")
take_care_of_nan(loans_df)
print("Nettoyage des NANs : DONE")

# Séparation du DataFrame en work et exam
loans_work_df, loans_exam_df =  work_exam_split(loans_df)

# Récupération des variables les plus importantes (j'ai un petit CPU)
all_cols_for_model = [ col for col in loans_work_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
frequ_transformed_loans_work_df = pd.DataFrame( MinMaxScaler().fit_transform(loans_work_df[all_cols_for_model]) , columns=all_cols_for_model ) 
select_kbest = SelectKBest(chi2, k=100)
select_kbest.fit_transform(frequ_transformed_loans_work_df, loans_work_df["TARGET"])
cols_to_keep = select_kbest.get_feature_names_out().tolist()

# Prendre toutes les variables
print("Nombre de colonnes retenues pour le modèle =",str(len(cols_to_keep)))
X = loans_work_df[cols_to_keep]
y = loans_work_df["TARGET"]

# Echantillonnage 
C_NB_SAMPLE = 10000
print("Echantillonnage à", C_NB_SAMPLE, "individus")
random_integers_for_sample = sample_without_replacement(n_population=len(y), n_samples=C_NB_SAMPLE)
X = X.iloc[random_integers_for_sample]
y = y.iloc[random_integers_for_sample]


# Séparation données TRAIN et TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# SMOTE
print("Smote Resampling : BEGIN")
smote_enn = SMOTEENN(sampling_strategy=0.8)
X_train, y_train = smote_enn.fit_resample(X_train, y_train)
print("Smote Resampling : DONE")

# Entrainement du modèle
model = make_pipeline(StandardScaler(),
                      LGBMClassifier(max_depth=8) )

print("Entrainement du modèle : BEGIN")
model.fit(X_train, y_train)
print("Entrainement du modèle : DONE")

# Ecriture (dump) du modèle par dans le fichier model.sav
output_filename = "model/dumps/boost_model.sav"

with open(output_filename, 'wb') as file_of_the_model:
    pickle.dump(model, file_of_the_model)

print("Écrit avec succès le modèle dans le fichier",output_filename)


# Entrainement du Lime Explainer
print("Entrainement du Lime Explainer : BEGIN")
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['crédit remboursé', 'défaut de paiement'],
    mode='classification'
)
print("Entrainement du Lime Explainer : DONE")

# Ecriture (dump) du Lime Table Explainer dans le fichier lime_tabular_explainer.sav
output_filename = "model/dumps/lime_tabular_explainer.sav"
with open(output_filename, 'wb') as f:
    dill.dump(lime_explainer, f)

print("Écrit avec succès le Lime Explainer dans le fichier",output_filename)




