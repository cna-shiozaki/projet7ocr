import os
import matplotlib.pyplot as plt
import pandas as pd


os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
loans_df = pd.read_csv('../data/output/loans.csv', nrows=None )

train_test_boundaries_list = list(loans_df.query( "index == 0 ").index)

assert len( train_test_boundaries_list) == 2

loans_work_df = loans_df.iloc[0:train_test_boundaries_list[1]]
loans_exam_df = loans_df.iloc[train_test_boundaries_list[1]:]

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

loans_train_df, loans_test_df =  train_test_split(loans_work_df, train_size=0.8)


X = loans_train_df[["AMT_CREDIT","AMT_INCOME_TOTAL"]].values
y = loans_train_df["TARGET"].values

clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, max_iter=5000))
clf.fit(X, y)

# Oh, shit. Ca ne marche pas du tout.
clf.predict([[406597.5, 202500.0]])
clf.predict(loans_test_df[["AMT_CREDIT","AMT_INCOME_TOTAL"]])

# plt.scatter(loans_train_df["AMT_CREDIT"],loans_train_df["AMT_INCOME_TOTAL"], c=loans_train_df["TARGET"],cmap='Set1')
