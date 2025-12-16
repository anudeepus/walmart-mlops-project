from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import uvicorn

app = FastAPI(title="Walmart Sales Prediction API")
mlflow.set_tracking_uri("http://0.0.0.0:5000")

class SalesInput(BaseModel):
    Store: int
    Holiday_Flag: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float

def predict_helper(model_name, data):
    try:
        # Load model version 1
        model = mlflow.sklearn.load_model(f"models:/{model_name}/1")
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        return {"model": model_name, "prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_model1")
def predict_linear(data: SalesInput):
    return predict_helper("Walmart_Linear_Reg", data)

@app.post("/predict_model2")
def predict_rf(data: SalesInput):
    return predict_helper("Walmart_Best_Model", data)

@app.post("/predict_model3")
def predict_gb(data: SalesInput):
    return predict_helper("Walmart_Gradient_Boost", data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
