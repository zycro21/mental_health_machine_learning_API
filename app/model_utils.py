# model_utils.py
import joblib
import pandas as pd

model = joblib.load("best_model_catboost.pkl")

def categorize_prediction(prediction: float) -> str:
    if prediction <= 1.0:
        return "Rendah"
    elif prediction <= 3.0:
        return "Sedang"
    else:
        return "Tinggi"

def predict_depression(data: dict) -> dict:
    df = pd.DataFrame([{
        "schizophrenia_share": data["schizophrenia_share"],
        "anxiety_share": data["anxiety_share"],
        "bipolar_share": data["bipolar_share"],
        "eating_share": data["eating_disorder_share"],
        "depression_dalys": data["DALYs"],  # Asumsikan DALYs ini untuk depression
        "schizophrenia_dalys": data["schizophrenia_dalys"],
        "bipolar_dalys": data["bipolar_dalys"],
        "eating_dalys": data["eating_dalys"],
        "anxiety_dalys": data["anxiety_dalys"],
        "DALYs": data["DALYs"],
        "suicide_rate": data["suicide_rate"]
    }])

    prediction = float(model.predict(df)[0])
    category = categorize_prediction(prediction)

    return {
        "resultLabel": category,
        "probabilityScore": round(prediction, 4)
    }
