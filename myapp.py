from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import pickle

app = FastAPI()

#Definir le format des input et output
class Model_Input(BaseModel):
    idClient: int
        
class Model_Target(BaseModel):
    resultat: float

# Où se trouvent les données et le modèle        
df = pd.read_csv('dataReduced2.csv')
 
model = pickle.load(open('model.pkl', 'rb'))

# L'app !
@app.post("/prediction", response_model=Model_Target)
def execute_prediction(input: Model_Input):
    idclient2 = input.__dict__['idClient']
    dfclient = df.loc[df['SK_ID_CURR'] == idclient2]
    dfclient = dfclient.drop(columns =['TARGET','SK_ID_CURR'])  
    res = model.predict_proba(dfclient)
    res = res[:,0]
    
    return {'resultat': res}

