import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
loans_df = pd.read_csv('../data/output/loans.csv', nrows=None )


from clean_functions import clean_up_nans

clean_up_nans(loans_df, "merge_with_bureau", ["BURO_","ACTIVE_","CLOSED_"])
clean_up_nans(loans_df, "merge_with_prev", ["PREV_","APPROVED_","REFUSED_"])
clean_up_nans(loans_df, "merge_with_pos", ["POS_"])
clean_up_nans(loans_df, "merge_with_ins", ["INSTAL_"])
clean_up_nans(loans_df, "merge_with_cc", ["CC_"])


# Remplacer par Zéro certaines Na (/TODO cela devrait plutôt être un préprocessing step de la pipeline)
cols = ["DAYS_EMPLOYED",                    # How many days before the application the person started current employment
        "AMT_REQ_CREDIT_BUREAU_DAY"         # Number of enquiries to Credit Bureau about the client one day before application
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_YEAR"
]

# /TODO Changer ça / mettre de plus de finesse à remplacer les Nan survivants par des zéros
loans_df.fillna( value=0, inplace=True )

# Pas trop sûr de comment des infinity sont arrivés là (INSTAL_PAYMENT_PERC_MAX)
#loans_df.loc[:,"INSTAL_PAYMENT_PERC_MAX"] = loans_df["INSTAL_PAYMENT_PERC_MAX"].replace(to_replace={np.inf: 0, -np.inf: 0 }, value=None)
cols_withs_infs_series = pd.DataFrame(np.isinf(loans_df.values), columns=loans_df.columns).sum()
cols_to_drop = list(cols_withs_infs_series.loc[ cols_withs_infs_series > 0 ].index)
loans_df.drop(labels=cols_to_drop,axis="columns",inplace=True)

# Il n'y a plus de Na !!
assert sum(list(dict(loans_df.isna().sum()).values())) == 0

# Séparation données de test et données d'entrainement
train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)

assert len( train_test_boundaries_list) == 2

loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]].copy()
loans_exam_df = loans_df.iloc[train_test_boundaries_list[1]:]



##############################
### FEATURE SELECTION ########
##############################
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

all_cols_for_model = [ col for col in loans_work_df.columns if col not in ("index","SK_ID_CURR","TARGET") ] 

select_kbest = SelectKBest(k=20)
select_kbest.fit_transform(loans_work_df[all_cols_for_model], loans_work_df["TARGET"])
select_kbest.get_feature_names_out()


minMaxScaler = MinMaxScaler()
frequ_transformed_loans_work_df = pd.DataFrame( minMaxScaler.fit_transform(loans_work_df[all_cols_for_model]) , columns=all_cols_for_model ) 
select_kbest = SelectKBest(chi2, k=20)
select_kbest.fit_transform(frequ_transformed_loans_work_df, loans_work_df["TARGET"])
select_kbest.get_feature_names_out()


# Train-Test split
loans_train_df, loans_test_df =  train_test_split(loans_work_df, train_size=0.8)


cols_to_keep = select_kbest.get_feature_names_out().tolist()

train_df = loans_train_df[["TARGET"] + cols_to_keep].copy()
train_df_sample = train_df.sample(20000)

X = train_df_sample[cols_to_keep].values
y = train_df_sample["TARGET"].values


from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 




print("### SVM Linear #####")
clf = make_pipeline(StandardScaler(),
                    SVC(random_state=0, kernel='linear',  probability=True  ))
clf.fit( X_resampled  , y_resampled)

y_pred = clf.predict(loans_test_df[cols_to_keep].values)
y_probas = clf.predict_proba(loans_test_df[cols_to_keep].values)

# Afficher la précision (accuracy)
print("Précision = ", metrics.accuracy_score(loans_test_df["TARGET"], y_pred))
print("Rappel = ", metrics.recall_score(loans_test_df["TARGET"], y_pred))

# 1/ Calculer la courbe ROC
fp_rate, tp_rate, thresholds = metrics.roc_curve(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1], pos_label=None)

# 2/ Calculer l'aire sous la courbe
auc = metrics.roc_auc_score(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1])
print("Aire sous la courbe =",str(round(auc,2)))

# 3/ Affichier la courbe ROC
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fp_rate,tp_rate,label="data 1, auc="+str(auc))
plt.legend(loc=4)



print("### SVM RBF #####")
clf = make_pipeline(StandardScaler(),
                    SVC(random_state=0, kernel='rbf',  probability=True  ))
clf.fit( X_resampled  , y_resampled)

y_pred = clf.predict(loans_test_df[cols_to_keep].values)
y_probas = clf.predict_proba(loans_test_df[cols_to_keep].values)

# Afficher la précision (accuracy)
print("Précision = ", metrics.accuracy_score(loans_test_df["TARGET"], y_pred))
print("Rappel = ", metrics.recall_score(loans_test_df["TARGET"], y_pred))

# 1/ Calculer la courbe ROC
fp_rate, tp_rate, thresholds = metrics.roc_curve(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1], pos_label=None)

# 2/ Calculer l'aire sous la courbe
auc = metrics.roc_auc_score(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1])
print("Aire sous la courbe =",str(round(auc,2)))

# 3/ Affichier la courbe ROC
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fp_rate,tp_rate,label="data 1, auc="+str(auc))
plt.legend(loc=4)



print("### RANDOM FOREST #####")
clf = make_pipeline(StandardScaler(),
                    RandomForestClassifier(random_state=0))
clf.fit( X_resampled  , y_resampled)

y_pred = clf.predict(loans_test_df[cols_to_keep].values)
y_probas = clf.predict_proba(loans_test_df[cols_to_keep].values)

# Afficher la précision (accuracy)
print("Précision = ", metrics.accuracy_score(loans_test_df["TARGET"], y_pred))
print("Rappel = ", metrics.recall_score(loans_test_df["TARGET"], y_pred))

# 1/ Calculer la courbe ROC
fp_rate, tp_rate, thresholds = metrics.roc_curve(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1], pos_label=None)

# 2/ Calculer l'aire sous la courbe
auc = metrics.roc_auc_score(y_true=loans_test_df["TARGET"], y_score=y_probas.T[1])
print("Aire sous la courbe =",str(round(auc,2)))

# 3/ Affichier la courbe ROC
plt.plot([0,1],[0,1],color='orange', linestyle='dashed')
plt.plot(fp_rate,tp_rate,label="data 1, auc="+str(auc))
plt.legend(loc=4)