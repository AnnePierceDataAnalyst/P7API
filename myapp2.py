from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import pickle

app = FastAPI()

#Definir le format des input et outputs
class Model_Input(BaseModel):
    idClient: int
        
class Model_Target1(BaseModel):
    resultat: float

#class Model_Target2(BaseModel):
    #importance: float
        
# Où se trouvent les données et le modèle        
df = pd.read_csv('dataReduced2.csv')
 
model = pickle.load(open('model.pkl', 'rb'))

# 1er endpoint
@app.post("/prediction", response_model=Model_Target1)
def execute_prediction(input: Model_Input):
    idclient2 = input.__dict__['idClient']
    dfclient = df.loc[df['SK_ID_CURR'] == idclient2]
    dfclient = dfclient.drop(columns =['TARGET','SK_ID_CURR'])  
    res = model.predict_proba(dfclient)
    res = res[:,0]
    
    return {'resultat': res}

# 2e endpoint
#@app.post("/importance", response_model=Model_Target2)
#def calculate_importance(input: Model_Input):
    
    #return {'importance': res}

