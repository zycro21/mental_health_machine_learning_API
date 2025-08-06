from fastapi import FastAPI, HTTPException
from schemas import PredictionInput
from model_utils import predict_depression

app = FastAPI()

# Endpoint default: GET /
@app.get("/")
def read_root():
    return {"message": "Akses endpoint berhasil"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        pred = predict_depression(input_data.dict())
        return pred  # ← langsung kembalikan dict dari model_utils
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {e}")
