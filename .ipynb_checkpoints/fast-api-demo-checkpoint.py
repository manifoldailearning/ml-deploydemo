from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle

with open("churn-model.bin", 'rb') as f:
   dv,model =  pickle.load(f) # deserialization

app = FastAPI()


class UserData(BaseModel):
    customerid : str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

@app.get("/")
def index():
    return {"message": "Hello World from get method"}

@app.get("/example")
def mfunc1():
    return {"message": "Hello from example path"}

@app.post("/predict")
def generate_predictions(churn_details:UserData): # parsed in the required
    data = churn_details.model_dump() # python dict
    X = dv.transform([data]) # transform
    y_pred = model.predict_proba(X)[:, 1] # generate prediction
    return {"Prediction":y_pred[0]} # return the prediction
    
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    