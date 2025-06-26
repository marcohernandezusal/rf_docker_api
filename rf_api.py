from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from contextlib import asynccontextmanager



class synth_reg(BaseModel):
    data: List[float] = Field(..., min_items=10, max_items=10)
    
model = None  # Placeholder for the model, to be loaded later

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Load the pre-trained model
        model = load("random_forest_model.pkl")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

app = FastAPI(title="Random Forest API", description="API for Random Forest model predictions", version="1.0", lifespan=lifespan)
    
@app.post("/predict", response_model=List[float])
async def predict(data: synth_reg):
    """
    Predict using the Random Forest model.

    Parameters:
    - data: List containing feature values.

    Returns:
    - List of predicted values.
    """
    global model
    if model is None:
        raise RuntimeError("Model is not loaded.")
    
    # Convert input data to a format suitable for prediction (2D array)
    # Reshape to (1, n_features) since we're predicting for one sample
    input_data = [data.data]  # This creates [[1.0, 2.0, ..., 10.0]]
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions.tolist()

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Random Forest API!"}