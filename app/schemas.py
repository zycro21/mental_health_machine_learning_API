# schemas.py
from pydantic import BaseModel

class PredictionInput(BaseModel):
    schizophrenia_share: float
    anxiety_share: float
    bipolar_share: float
    eating_disorder_share: float
    DALYs: float
    suicide_rate: float
    depression_dalys: float
    schizophrenia_dalys: float
    bipolar_dalys: float
    eating_dalys: float
    anxiety_dalys: float
