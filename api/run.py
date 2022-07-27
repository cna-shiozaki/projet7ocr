import os, io
import uvicorn
import traceback
import logging

from typing import Union
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from api.projet7_server import Projet7Server


app = FastAPI()


# Monter la Single-Page application buildée sous /webapp
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "../spa/dist")
#st_abs_file_path = os.path.join(script_dir, "../spa/webapp")
app.mount("/webapp", StaticFiles(directory=st_abs_file_path), name="static")

# Instancier le Prediction Engine, et le préparer
projet7Server = Projet7Server()
projet7Server.initialize()



@app.get("/")
def read_root():
    return {"Racine": "Bonjour Amigo"}


########################################################################
#### END-POINT : Obtenir la prédiction + le détail d'un client  ########
########################################################################
@app.get("/loans/{loan_id}")
def get_single_loan_data(loan_id: int, q: Union[str, None] = None):
    try:
        prediction, predicted_score = projet7Server.fetch_customer_prediction(loan_id=loan_id)
        details = projet7Server.fetch_customer_details(loan_id=loan_id)
        
        dico_resp = {
            "prediction" : prediction, 
            "customer_score" : predicted_score } 
        dico_resp.update(details)
        
        return dico_resp
    
    except Exception as e:
        return { "error" : traceback.format_exc() }

########################################################################
#### END-POINT : Numéro de prêt aléatoire  #############################
########################################################################
@app.get("/random_loan_id/")
def get_random_loan_id():
    try:
        return str(projet7Server.fetch_random_loan_id())
    except Exception as error:
        return { "error" : error }

########################################################################
#### END-POINT : LIME TABULAR EXPLAINER  ###############################
########################################################################
@app.get("/lime_tabular_explain/{loan_id}")
def get_tabular_explain(loan_id : int):
    try:
         return projet7Server.fetch_lime_tabular(loan_id)
        #image = projet7Server.fetch_random_loan_id()
    except Exception as error:
        return { "error" : error }


########################################################################
#### END-POINT : /TODO IMAGE   #############################
########################################################################
@app.get("/shap_force/{loan_id}")
def get_shap_force(loan_id : int):
    try:
        #os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work")
        with open("shap_stuff.png", "rb") as file:
            img = file.read()
            return StreamingResponse(io.BytesIO(img), media_type="image/png")

    except Exception as error:
        return { "error" : error }

########################################################################
#### END-POINT : BOX PLOT COMPARISON   #################################
########################################################################
@app.get("/boxplot/{loan_id}")
def get_box_plot(loan_id : int):
    try:
        buffer = projet7Server.fetch_box_plot(loan_id)        
        return StreamingResponse(buffer, media_type="image/png")

    except Exception as error:
        return { "error" : error }


########################################################################
#### END-POINT : READ SPECIFIC FIELD  ##################################
########################################################################
@app.get("/specific_field/{loan_id}/{specific_field}")
def get_specific_field(loan_id : int, specific_field : str):
    try:
         return projet7Server.fetch_specific_field(loan_id, specific_field)
    except Exception as error:
        return { "error" : error }


########################################################################
#### END-POINT : ALL COLUMNS NAMES  ####################################
########################################################################
@app.get("/all_columns_names/")
def get_all_columns_names():
    try:
         return projet7Server.fetch_column_names()
    except Exception as error:
        return { "error" : error }

@app.get("/exemple/")
def get_exemple():
    return {"Juste un test": 3 }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)