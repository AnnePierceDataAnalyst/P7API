from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import pickle
import lime

app = FastAPI()

#Definir le format des input et output
class Model_Input(BaseModel):
    idClient: int
        
class Model_Target(BaseModel):
    resultat: float
    features: list
    importance: list 

# Où se trouvent les données et le modèle        
df = pd.read_csv('dataReduced2.csv')
 
model = pickle.load(open('model.pkl', 'rb'))

# L'endpoint
@app.post("/prediction", response_model=Model_Target)
def execute_prediction(input: Model_Input):
    
    # calculer score
    idclient2 = input.__dict__['idClient']
    dfclient = df.loc[df['SK_ID_CURR'] == idclient2]
    dfclient = dfclient.drop(columns =['TARGET','SK_ID_CURR'])  
    res = model.predict_proba(dfclient)
    res = res[:,0]
    
    # calculer importances avec Lime
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]
    X = df[feats].values 
    
    explainer = lime.lime_tabular.LimeTabularExplainer(X, mode="classification",
                                              feature_names=feats)
    
    idx = df.index[df["SK_ID_CURR"] == idclient2]
    exp = explainer.explain_instance(X[idx][0], model.predict_proba)
    
    # Créer liste de features
    features = []
    for i in range(5):
        features.append (exp.as_list()[i][0])
        
    # Créer liste d'importances
    importance = []
    for i in range(5):
        importance.append (exp.as_list()[i][1])        

    
    return {'resultat': res, 'features':features,'importance': importance}

