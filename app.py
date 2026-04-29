from fastapi import FastAPI
import mlflow.sklearn
import numpy as np


app = FastAPI()

model = mlflow.sklearn.load_model("models:/titanic_model/latest")

@app.get("/predict")
def predict(pclass:int,sex:int,age:float,fare:float):
    """
    pclass: 1,2,3
    sex: 0 = male, 1 = female
    age: float
    fare: float
    """
    features = np.array([[pclass,sex,age,fare]])
    
    prediction = model.predict(features)[0]
    
    return {
        "prediction" : int(prediction),
        "result" : "Survived" if prediction==1 else "Not survived"
    }