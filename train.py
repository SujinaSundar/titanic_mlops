from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import pickle
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("titanic exp")

df = pd.read_csv("data/titanic.csv")

X = df.drop('Survived',axis=1)
Y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

with mlflow.start_run():
    model = RandomForestClassifier()
    
    model.fit(x_train,y_train)

    
    accuracy = model.score(x_test,y_test)
    
    mlflow.log_param("n_estimators",100)
    mlflow.log_metric("Accuracy",accuracy)
    mlflow.sklearn.log_model(model,registered_model_name="titanic_model")
    
    print("Accuracy",accuracy)
