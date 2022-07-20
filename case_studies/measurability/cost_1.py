# Combien rapporte un client ?
# Combien coûte un client qui ne rembourse pas son prêt ?


import os
import matplotlib.pyplot as plt
import pandas as pd


###################################
### RECUPERATION DES DONNEES ######
###################################
os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
application_train_df = pd.read_csv('../data/input/application_train.csv', nrows=None )
previous_application_df = pd.read_csv('../data/input/previous_application.csv', nrows=None )
credit_card_balance_df = pd.read_csv('../data/input/credit_card_balance.csv', nrows=None )
installments_payments_df = pd.read_csv('../data/input/installments_payments.csv', nrows=None )
pos_cash_balance_df = pd.read_csv('../data/input/POS_CASH_balance.csv', nrows=None )

bureau_df = pd.read_csv('../data/input/bureau.csv', nrows=None )
bureau_balance_df = pd.read_csv('../data/input/bureau_balance.csv', nrows=None )



good_cols = ["SK_ID_PREV","SK_ID_CURR"	,"NAME_CONTRACT_TYPE","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_GOODS_PRICE","NAME_CONTRACT_STATUS","CHANNEL_TYPE"]
previous_application_df.loc[ previous_application_df["SK_ID_CURR"] == 100005 ][good_cols]



# Pour se convaincre que credit_card_balance ne contient que les paiements des crédits de type "Revolving Loan"
pd.merge(left=previous_application_df, right=credit_card_balance_df, on="SK_ID_PREV")["NAME_CONTRACT_TYPE"].unique()


# 'installments_payments_df' contient aussi bien des crédits de type "Consumer Loan" /  "Cash Loan" / "Revolving Loan"
prev_app_installment = pd.merge(left=previous_application_df, right=installments_payments_df, on="SK_ID_PREV")["NAME_CONTRACT_TYPE"]
prev_app_installment["NAME_CONTRACT_TYPE"].value_counts()

# POS_CASH_balance ne contient que "Consumer Loan" /  "Cash Loan"
prev_pos_cash_balance = pd.merge(left=previous_application_df, right=pos_cash_balance_df, on="SK_ID_PREV")
prev_pos_cash_balance["NAME_CONTRACT_TYPE"].value_counts()


# USEFUL : Ce que le client 100005 a fait pour son Consumer Loan 2495675
# (Il y a une entrée 'Signed', suivie de n 'Actives' qui se termine enfin par 'Completed')
pos_cash_balance_df.loc[ pos_cash_balance_df["SK_ID_CURR"] == 100005 ]

installments_payments_df.loc[ installments_payments_df["SK_ID_CURR"] == 100005 ]

# Somme payée par le client 100005 : 56161.845 (à comparer avec un AMT_CREDIT de 40153.5)
installments_payments_df.loc[ installments_payments_df["SK_ID_CURR"] == 100005 ]["AMT_PAYMENT"].sum()


# Problemo, que faire des revolving loans ?
previous_application_df.query( "NAME_CONTRACT_TYPE == 'Revolving loans'")


# Je sais !
# Pas de soucis en fait, il suffit toujours d'aller regarder dans installments_payments_df que tous les paiement prévus ont été fait en temps et en heure. 
# Voir ce qui était prévu, et ce qui a été payé
installments_payments_df.query( "SK_ID_CURR == 142748")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()

# Voir quelques entrées de prêts revolving de vêtements
previous_application_df.query( "(NAME_CONTRACT_TYPE == 'Revolving loans') & (NAME_SELLER_INDUSTRY == 'Clothing')")[["SK_ID_PREV","SK_ID_CURR","NAME_CONTRACT_TYPE","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","NAME_SELLER_INDUSTRY"]]



# Essaier de comprendre ça
installments_payments_df.query( "SK_ID_CURR == 201148")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()  # Pas la même !!

installments_payments_df.query( "SK_ID_PREV == 2056408")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()  # Le mec n'a rien emprunté de ce qu'il pouvait !

good_cols = ["SK_ID_PREV","SK_ID_CURR"	,"NAME_CONTRACT_TYPE","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_GOODS_PRICE","NAME_CONTRACT_STATUS","CHANNEL_TYPE"]
previous_application_df.loc[ previous_application_df["SK_ID_CURR"] == 201148 ][good_cols]


# Et pour le consumer loan 1128117, quelqu'un m'explique pourquoi le gus n'a emprunté que 382 783 et qu'il a rembourné 818 017...? (alors que les installments prévoyaient 430 335) ?
# (alors que l'annuité devait être de 27 009 )
installments_payments_df.query( "SK_ID_PREV == 1128117	")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()  # Pas la même !!


# Prenons de nouvelles données
application_train_df.loc[2190:2200]  # Etudions le loan 102571
good_cols = ["SK_ID_PREV","SK_ID_CURR","NAME_CONTRACT_TYPE","NAME_CONTRACT_STATUS","AMT_ANNUITY","AMT_APPLICATION","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_GOODS_PRICE","CHANNEL_TYPE"]
previous_application_df.loc[ previous_application_df["SK_ID_CURR"] == 102571 ][good_cols]

installments_payments_df.query( "SK_ID_PREV == 1205044	")
installments_payments_df.query( "SK_ID_PREV == 1205044	")[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()



