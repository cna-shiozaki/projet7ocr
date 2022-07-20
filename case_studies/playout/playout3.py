import os


os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")

from api.projet7_server import Projet7Server


projet7_server = Projet7Server()

projet7_server.initialize()



try:
    loan_id = 100001
    predicted_score = projet7_server.fetch_customer_prediction(loan_id=loan_id)
    details = projet7_server.fetch_customer_details(loan_id=loan_id)
    random = projet7_server.fetch_random_loan_id()
except Exception as error:
    print(error)


### TRES IMPORTANT - mettre l'import !!
os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
import pickle
from model.baseline import BaselineModel
os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\model\dumps")
file_of_the_model = open("baseline_model.sav","rb")
baseline_model = pickle.load(file_of_the_model)
file_of_the_model.close()
print(baseline_model.predict(["num1","num2","num3"]))
