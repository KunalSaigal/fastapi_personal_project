from pydantic import BaseModel, Field
from typing import Annotated, Literal

class PredictionResponse(BaseModel):
    prediction: Annotated[Literal['Low', 'Medium', 'High'], Field(..., description="The predicted indsurance premium category")]
    confidence: Annotated[float, Field(..., description="Model's confidence score for predicted category Range(0,1)")]
    class_probability: Annotated[dict[Literal['Low', 'Medium', 'High'],float], Field(..., description="Probability of each category")]