# Combien rapporte un client ?
# Combien coûte un client qui ne rembourse pas son prêt ?


import os, sys
import matplotlib.pyplot as plt
import pandas as pd


###################################
### RECUPERATION DES DONNEES ######
###################################
os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\model")

# Chargement du dataset
previous_application_df = pd.read_csv('../data/input/previous_application.csv', nrows=None )
installments_payments_df = pd.read_csv('../data/input/installments_payments.csv', nrows=None )

# Informations génériques
if "--display" in sys.argv  :
    previous_application_df.loc[previous_application_df.NAME_CONTRACT_TYPE == "Cash loans"]["AMT_CREDIT"].describe()


# Date prévue de fin de prêt : supprimer les NaN
previous_application_df.dropna(axis="index", how="any", subset=["DAYS_TERMINATION"],inplace=True)

# Date prévue de fin de prêt : supprimer les loans qui ne sont pas encore finis (ou ceux qui ont fini dans les 2 mois précédents)
previous_application_df.drop(index=previous_application_df.query("DAYS_TERMINATION > -60").index, inplace=True)

# Somme des paiements effectués par les clients (pour rembourser leur crédit)
repayments_of_credit = installments_payments_df.groupby(["SK_ID_PREV"])[["AMT_INSTALMENT","AMT_PAYMENT"]].sum()

# Aggrégation des sommes payées vs somme empruntée
useful_cols = ["SK_ID_PREV","SK_ID_CURR","NAME_CONTRACT_TYPE","AMT_CREDIT","AMT_DOWN_PAYMENT","AMT_INSTALMENT","AMT_PAYMENT"]
results = pd.merge(left=previous_application_df, right=repayments_of_credit, left_on="SK_ID_PREV", right_index=True)[useful_cols]

# Calcul de l'écart entre argent prêté et argent reçu
results["balance"] = results["AMT_PAYMENT"] - results["AMT_CREDIT"]

profit = results.query("balance > 0 ")
loss = results.query("balance < 0 ")

# Il n'y a que 0,15% des clients qui ont été totalement insolvables, et à qui on n'a pas pu recouvrer l'argent
print(str(len(profit)), " emprunts remboursés")
print(str(len(loss)), " emprunts non remboursés")

print("Un emprunt remboursé remporte souvent",int(profit["balance"].median()))
print("Un emprunt non remboursé fait perdre souvent",abs(int(loss["balance"].median())) )

if "--display" in sys.argv  : 
    plt.title("Estimation des profits")
    plt.hist(profit["balance"],bins=10000, color="limegreen")
    plt.xlim(0,200000)
    plt.xlabel("Emprunts")
    plt.ylabel("Profit gagné")
    plt.show()

    plt.title("Estimation des pertes")
    plt.hist(abs(loss["balance"]),bins=100, color="tomato")
    plt.xlim(0,2000000)
    plt.xlabel("Emprunts")
    plt.ylabel("Perte")