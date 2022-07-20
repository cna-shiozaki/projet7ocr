# Faire fonctioner un objet projet7Server


from api.projet7_server import Projet7Server

projet7Server = Projet7Server()
projet7Server.initialize()


res1 = projet7Server.fetch_customer_details(100001)
res2 = projet7Server.fetch_customer_prediction(100001)


try:
    predicted_score = projet7Server.fetch_customer_prediction(loan_id=100002)
    details = projet7Server.fetch_customer_details(loan_id=100002)
    
    dico_resp = {"customer_score" : predicted_score } 
    dico_resp.update(details)

except Exception as error:
    errordict = { "error" : str(error) }



stroo = "requête finie"