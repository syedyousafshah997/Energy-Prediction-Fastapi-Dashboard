"""
FastAPI Dashboard for Steel Industry Energy Consumption prediction.

target is `Usage_kWh`, categorical columns are one-hot encoded inside the saved pipeline.

Run with:
   uvicorn main:app --reload
   
"""

import os
import joblib
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Steel Industry Energy Consumption Dashboard")

MODEL_PATH = "model.joblib"
DATA_PATH = "Steel_energy_consumption_engineered.csv"
STATIC_DIR = "static"
TARGET_COL = "Usage_kWh"
DROP_COLS = ["date", "High_Load"]   # same drop as Week 2

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Load model pipeline and reference data once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

df_reference = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
FEATURE_NAMES = [c for c in df_reference.columns if c != TARGET_COL]

# categorical columns get a dropdown in the form; numeric columns get a text box
CATEGORICAL_OPTIONS = {
    col: sorted(df_reference[col].unique().tolist())
    for col in df_reference.select_dtypes(include="object").columns
    if col != TARGET_COL
}



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/predict")
def predict_form(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "features": FEATURE_NAMES,
            "categorical_options": CATEGORICAL_OPTIONS,
            "prediction": None,
        },
    )


@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()

    input_data = {}
    for feature in FEATURE_NAMES:
        raw_value = form_data.get(feature, "")
        if feature in CATEGORICAL_OPTIONS:
            input_data[feature] = raw_value
        else:
            try:
                input_data[feature] = float(raw_value)
            except ValueError:
                input_data[feature] = 0.0

    input_df = pd.DataFrame([input_data])[FEATURE_NAMES]
    prediction = round(float(model.predict(input_df)[0]), 3)

    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "features": FEATURE_NAMES,
            "categorical_options": CATEGORICAL_OPTIONS,
            "prediction": prediction,
        },
    )
