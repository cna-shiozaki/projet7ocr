import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
from etl.clean_up import take_care_of_nan, work_exam_split


# Chargement du dataset
loans_df = pd.read_csv('data/output/loans.csv', nrows=None )

# Retirer les NaN
take_care_of_nan(loans_df)

# Séparation du DataFrame en work et exam
loans_work_df, loans_exam_df =  work_exam_split(loans_df)


##############################################################
############## PROBLEME AVEC MIN MAX SCALER ##################
##############################################################

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

my_df1 = loans_work_df[loans_work_df.columns[0:10]].copy()
my_df2 = loans_work_df[loans_work_df.columns[0:10]].copy()
my_df3 = loans_work_df[loans_work_df.columns[0:10]].copy()
my_df4 = loans_work_df[loans_work_df.columns[0:10]].copy()

my_df2.loc[:, "AMT_INCOME_TOTAL"] = my_df2["AMT_INCOME_TOTAL"] / my_df2["AMT_INCOME_TOTAL"].max()
minMaxScaler_3 = MinMaxScaler() ; minMaxScaler_4 = MinMaxScaler()
my_df3.loc[:, "AMT_INCOME_TOTAL"] = minMaxScaler_3.fit_transform(loans_work_df[["AMT_INCOME_TOTAL"]]).T[0]
my_df4 = pd.DataFrame( minMaxScaler_4.fit_transform(my_df4), columns=my_df4.columns)


select_kbest1 = SelectKBest(chi2, k=3)
all_cols_for_model = [ col for col in my_df1.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
select_kbest1.fit_transform(my_df1[all_cols_for_model], my_df1["TARGET"])

select_kbest2 = SelectKBest(chi2, k=3)
all_cols_for_model = [ col for col in my_df2.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
select_kbest2.fit_transform(my_df2[all_cols_for_model], my_df2["TARGET"])

select_kbest3 = SelectKBest(chi2, k=3)
all_cols_for_model = [ col for col in my_df3.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
select_kbest3.fit_transform(my_df3[all_cols_for_model], my_df3["TARGET"])

select_kbest4 = SelectKBest(chi2, k=3)
all_cols_for_model = [ col for col in my_df4.columns if col not in ("index","SK_ID_CURR","TARGET") ] 
select_kbest4.fit_transform(my_df4[all_cols_for_model], my_df4["TARGET"])

pd.DataFrame(data=[
    list(select_kbest1.scores_) + [ str(select_kbest1.get_feature_names_out())], 
    list(select_kbest2.scores_) + [ str(select_kbest2.get_feature_names_out())], 
    list(select_kbest3.scores_) + [ str(select_kbest3.get_feature_names_out())],
    list(select_kbest4.scores_) + [ str(select_kbest4.get_feature_names_out())]] ,
columns=all_cols_for_model + ["Best v.a."], 
index=["chi2","chi 2 scale manuel","chi2 with MinMaxScaler","chi2 total MinMaxScaler"])
