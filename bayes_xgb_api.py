from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier
import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import joblib
import pandas as pd

app = FastAPI()

# Load the pre-trained model
model = XGBClassifier()
model.load_model('bayesopt_xgb_model.xgb')

# load preprocessing pkl
# 1) LabelEncoder
agent_le = joblib.load("agent_le.pkl")
map_le = joblib.load("map_le.pkl")

# 2) MinMaxScaler
minmax_scaler = joblib.load("minmax_scaler.pkl")

# 3) StandardScaler
standard_scaler = joblib.load("standard_scaler.pkl")

index = ['ct1_agent', 'ct1_R', 'ct1_ACS', 'ct1_KAST', 'ct1_ADR', 
         'ct2_agent','ct2_R', 'ct2_ACS', 'ct2_KAST', 'ct2_ADR', 
         'ct3_agent', 'ct3_R','ct3_ACS', 'ct3_KAST', 'ct3_ADR',
         'ct4_agent', 'ct4_R', 'ct4_ACS','ct4_KAST', 'ct4_ADR', 
         'ct5_agent', 'ct5_R', 'ct5_ACS', 'ct5_KAST','ct5_ADR',
         't1_agent', 't1_R', 't1_ACS', 't1_KAST', 't1_ADR',
         't2_agent', 't2_R', 't2_ACS', 't2_KAST', 't2_ADR', 
         't3_agent', 't3_R','t3_ACS', 't3_KAST', 't3_ADR', 
         't4_agent', 't4_R', 't4_ACS', 't4_KAST','t4_ADR', 
         't5_agent', 't5_R', 't5_ACS', 't5_KAST', 't5_ADR', 
         'map']

attr_list = ['ct1_ACS', 'ct1_ADR', 'ct1_R', 'ct2_ACS', 'ct2_ADR', 'ct2_R', 'ct3_ACS', 'ct3_ADR', 'ct3_R', 'ct4_ACS', 'ct4_ADR', 'ct4_R', 'ct5_ACS', 'ct5_ADR', 'ct5_R', 't1_ACS',
             't1_ADR', 't1_R', 't2_ACS', 't2_ADR', 't2_R', 't3_ACS', 't3_ADR', 't3_R', 't4_ACS', 't4_ADR', 't4_R', 't5_ACS', 't5_ADR', 't5_R']

kast_list = ['ct1_KAST','ct2_KAST','ct3_KAST','ct4_KAST','ct5_KAST',
             't1_KAST','t2_KAST','t3_KAST','t4_KAST','t5_KAST']

agent_list = ['ct1_agent', 'ct2_agent', 'ct3_agent', 'ct4_agent', 'ct5_agent',
              't1_agent', 't2_agent', 't3_agent', 't4_agent', 't5_agent']

class PredictionInput(BaseModel):
    data: List[str]

class PredictionOutput(BaseModel):
    ct_prediction: str
    ct_proba: float
    t_prediction: str
    t_proba: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput):
    try:
        input_data = input.data
        df = pd.DataFrame([input_data], columns=index)
        
        # clean data
        for col in attr_list:
            df[col] = df[col].astype(float)
        
        for col in kast_list:
            df[col] = df[col].astype('string')
            
            # convert percentage data to float
            df[col] = (pd.to_numeric(df[col].str.strip('%'), errors='coerce') / 100)
        
        # preprocessing input data
        df[kast_list] = standard_scaler.transform(df[kast_list])

        df[attr_list] = minmax_scaler.transform(df[attr_list])

        for agent in agent_list:
            df[agent] = agent_le.transform(df[agent])
        
        df["map"] = map_le.transform(df["map"])
        
        # Predict
        prediction = model.predict_proba(df.iloc[0].to_numpy().reshape(1, -1))
        
        # Return the prediction
        return PredictionOutput(ct_prediction=("win" if prediction[0][0] >= 0.5 else "lose"), 
                                ct_proba=prediction[0][0], 
                                t_prediction=("win" if prediction[0][1] >= 0.5 else "lose"), 
                                t_proba=prediction[0][1])
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)