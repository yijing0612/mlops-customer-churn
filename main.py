from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow.sklearn

# Start a run
with mlflow.start_run() as run:
    # Train your model as usual...
    model = 'randomforest_200_10_model.pkl'
    mlflow.sklearn.log_model(model, "model")

    # Register the model in the registry
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="RF_200_10"
    )

    print("Model registered! Check MLflow UI -> Models tab")

app = FastAPI()

class CustomerData(BaseModel):
    feature1: float
    feature2: float
    # Add all required features...

# Load model from MLflow
model = mlflow.sklearn.load_model("models:/RF_200_10/Production")

@app.post("/predict")
def predict(data: CustomerData):
    input_data = [[data.feature1, data.feature2]]  # Add all features in order
    prediction = model.predict(input_data)
    return {"churn_prediction": int(prediction[0])}