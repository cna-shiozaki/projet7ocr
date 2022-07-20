import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA

os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")

from etl.clean_up import take_care_of_nan, work_exam_split

# Chargement du dataset
loans_df = pd.read_csv('./data/output/loans.csv', nrows=None )

# Retirer les NaN
take_care_of_nan(loans_df)

# Séparation du DataFrame en work et exam
loans_work_df, loans_exam_df =  work_exam_split(loans_df)

X = loans_work_df[[ col for col in loans_work_df.columns if col not in ["TARGET","index","SK_ID_CURR"]  ]]

# Standard Scaling
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

# ACP à 99%
pca = PCA(n_components=0.99)
pca.fit(X)

cumlated = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(15,4))
plt.bar(x=np.arange(len(cumlated)),height=cumlated,color="coral")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance expliquée")
plt.title("Analyse en Composantes Principales")