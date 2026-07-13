import pandas as pd
from fastapi import FastAPI
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

app = FastAPI()

class EmployeeData(BaseModel):
    age: int
    educ_bac: int

try:
    df = pd.read_csv("employment_08_09.csv")
    df = df.dropna(subset=['earnwke'])
    X = df[['age', 'educ_bac']] 
    y = df['earnwke']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR(kernel='rbf')
    model.fit(X_scaled, y)
    model_ready = True
except Exception as e:
    model_ready = False
    error_msg = str(e)

@app.get("/")
def read_root():
    return {"message": "Destek Vektor Regresyonu (SVR) API'sine Hosgeldiniz!"}

@app.post("/predict")
def predict_salary(data: EmployeeData):
    if not model_ready:
        return {"error": "Model egitilemedi", "details": error_msg}
    
    input_df = pd.DataFrame([{"age": data.age, "educ_bac": data.educ_bac}])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    return {"predicted_weekly_earnings": round(prediction, 2)}
