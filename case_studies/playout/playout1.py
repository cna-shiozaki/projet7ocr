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

#########################################
### ETUDIER LE DATASET ####
## Variables qui sont souvent remplies ##
#########################################
import missingno as msno
msno.matrix(loans_train_df.sample(250))
msno.bar(loans_train_df.sample(250))

loans_train_sample_df = loans_train_df.sample(250)
count_of_na_series = loans_train_sample_df.isna().sum()
columns_almost_always_filled = list(count_of_na_series.loc[count_of_na_series < 50 ].index)

#########################################
### ETUDIER LE DATASET ####
## Variables corrélées deux à deux ##
#########################################
# Générons la matrice de correlation des v.a. deux à deux
import seaborn as sns
#loans_df_special = loans_df[[ col for col in loans_df if ( col != 'index' and col != 'TARGET' ) ]]
#loans_df_special = loans_df[[ col for col in loans_df.columns if ( col != 'index' ) ]]
loans_df_special = loans_train_df[[ col for col in loans_df.columns if ( col != 'index' ) and col in columns_almost_always_filled ]]

corr = loans_df_special.sample(1000).corr()
corr = corr.dropna(axis="index",how="all").dropna(axis="columns",how="all")
plt.figure(figsize=(15,15))
sns.heatmap(corr)

# Essayons de trouver des variables indépedantes corrélées à TARGET
corr_target = corr[["TARGET"]].sort_values(ascending=False,by="TARGET").head(10)
col_to_check = "CC_CNT_DRAWINGS_CURRENT_MAX"
corr[[col_to_check]].loc[ [ e for e in list(corr_target.index) if e !="TARGET"]  ].sort_values(ascending=False,by=col_to_check)

# Identification des colonnes à garder pour le modèle
cols_to_keep = [ "DAYS_BIRTH",  "PREV_NAME_CONTRACT_STATUS_Refused_MEAN"  , "BURO_DAYS_CREDIT_MEAN"  ]

# Suppression des NA
loans_train_df_without_na = loans_train_df[cols_to_keep + ["TARGET"]].dropna(axis="index",how="any")

##############################
### ENTRAINEMENT #############
##############################
X = loans_train_df_without_na[cols_to_keep].values
y = loans_train_df_without_na["TARGET"].values

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, max_iter=5000))
clf.fit(X, y)

##############################
### EVALUATION #############
##############################
loans_test_df_without_na = loans_test_df[cols_to_keep + ["TARGET"]].dropna(axis="index",how="any").sample(1000)

X_test = loans_test_df_without_na[cols_to_keep].values
y_test = loans_test_df_without_na["TARGET"].values
 
y_pred = pd.Series(clf.predict(X_test), name="Predicted")
y_test = pd.Series(y_test, name="Actual")

pd.crosstab(y_pred, y_test )

# plt.scatter(loans_test_df_without_na["DAYS_BIRTH"],loans_test_df_without_na["BURO_DAYS_CREDIT_MEAN"], c=loans_test_df_without_na["TARGET"],cmap='Set1')


