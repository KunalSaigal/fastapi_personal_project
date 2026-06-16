from pathlib import Path

from fastapi import FastAPI , HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ai_prediction.schema.user_input import UserModel
from ai_prediction.model.predict import MODEL_VERSION, model
import pandas as pd
from ai_prediction.model.predict import get_prediction
from ai_prediction.schema.prediction_reponse import PredictionResponse

app = FastAPI()

well_known_dir = Path(__file__).resolve().parent / ".well-known"
app.mount(
    "/.well-known",
    StaticFiles(directory=str(well_known_dir), html=False),
    name="well-known",
)

@app.get("/")
async def home():
    return {"message": "Welcome to the AI Prediction Model Demo"}

@app.get("/health")
async def health():
    return {"status": "ok", "version": MODEL_VERSION, "model_loaded": model is True,"message": "Model is healthy"}


@app.post("/predict",response_model=PredictionResponse)
async def predict(data: UserModel):
    try:
        prediction = get_prediction({
            'bmi':data.bmi, 
            'income_lpa':data.income_lpa,
            'age_group':data.age_group,
            'city_tier':data.city_tier,
            'occupation':data.occupation,
            "lifestyle_risk": data.lifestyle_risk
        })

        return JSONResponse(status_code=200, content={'response': prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})