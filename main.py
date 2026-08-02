import os
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads .env file locally; on Render, env vars are injected directly so this line just does nothing there

# ---------------------------------------------------------
# Always resolve paths relative to this file, not the CWD
# the process happens to be launched from.
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'Mental_Health_Model.pkl'))

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Groq client for the /explain narration step
# Set GROQ_API_KEY as an environment variable on Render
# ---------------------------------------------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------
# SHAP explainer setup
# We build this ONCE at startup (not per-request) because
# fitting/initializing an explainer is expensive.
#
# IMPORTANT: SHAP's default tabular masker calls np.isclose()
# under the hood, which only works on numeric data. Several of
# our features (Gender, Country, Academic_Level, etc.) are
# strings, which caused:
#   TypeError: unsupported operand type(s) for -: 'str' and 'str'
#
# Fix: label-encode all categorical columns into integers before
# handing anything to SHAP, and wrap model.predict so it decodes
# those integers back to strings before calling the real model.
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'Age', 'Gender', 'Country', 'Academic_Level', 'Most_Used_Platform',
    'Purpose_Of_Use', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours',
    'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Stress_Level', 'Grouped_country'
]

CATEGORICAL_COLUMNS = [
    'Gender', 'Country', 'Academic_Level', 'Most_Used_Platform',
    'Purpose_Of_Use', 'Stress_Level', 'Grouped_country'
]

try:
    # Preferred: use a real sample of training data for the background set.
    # Export this once from your notebook:
    # df.sample(50).to_csv('shap_background.csv', index=False)
    background_df = pd.read_csv(os.path.join(BASE_DIR, 'shap_background.csv'))[FEATURE_COLUMNS]
except FileNotFoundError:
    # Fallback: one plausible "average" row, duplicated a few times.
    # Includes a couple of varied rows so the label encoders below
    # don't choke on a single-category column.
    background_df = pd.DataFrame([
        {
            'Age': 21, 'Gender': 'Male', 'Country': 'India', 'Academic_Level': 'Undergraduate',
            'Most_Used_Platform': 'Instagram', 'Purpose_Of_Use': 'Entertainment',
            'Avg_Daily_Usage_Hours': 4.0, 'Daily_Unlocks': 50, 'Study_Hours': 3.0,
            'Physical_Activity_Hours': 1.0, 'Sleep_Hours_Per_Night': 7.0,
            'Stress_Level': 'Medium', 'Grouped_country': 'India'
        },
        {
            'Age': 24, 'Gender': 'Female', 'Country': 'USA', 'Academic_Level': 'Graduate',
            'Most_Used_Platform': 'YouTube', 'Purpose_Of_Use': 'Education',
            'Avg_Daily_Usage_Hours': 2.5, 'Daily_Unlocks': 30, 'Study_Hours': 5.0,
            'Physical_Activity_Hours': 2.0, 'Sleep_Hours_Per_Night': 8.0,
            'Stress_Level': 'Low', 'Grouped_country': 'USA'
        },
        {
            'Age': 19, 'Gender': 'Male', 'Country': 'Other', 'Academic_Level': 'High School',
            'Most_Used_Platform': 'TikTok', 'Purpose_Of_Use': 'Networking',
            'Avg_Daily_Usage_Hours': 6.0, 'Daily_Unlocks': 80, 'Study_Hours': 2.0,
            'Physical_Activity_Hours': 0.5, 'Sleep_Hours_Per_Night': 6.0,
            'Stress_Level': 'High', 'Grouped_country': 'Other'
        },
    ] * 10)

# Fit one LabelEncoder per categorical column on the background sample.
# (These only need to know the set of categories that can show up.)
encoders = {col: LabelEncoder().fit(background_df[col].astype(str)) for col in CATEGORICAL_COLUMNS}


def encode_row(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical string columns -> integer codes for SHAP."""
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = encoders[col].transform(df[col].astype(str))
    return df[FEATURE_COLUMNS]


def decode_row(arr) -> pd.DataFrame:
    """Convert integer codes from SHAP back into a real model input row."""
    df = pd.DataFrame(np.asarray(arr), columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            codes = df[col].round().astype(int).clip(
                lower=0, upper=len(encoders[col].classes_) - 1
            )
            df[col] = encoders[col].inverse_transform(codes)
        else:
            df[col] = pd.to_numeric(df[col])
    return df


def predict_encoded(X):
    """SHAP-facing predict function: takes numeric arrays, decodes, predicts."""
    return model.predict(decode_row(X))


background_encoded = encode_row(background_df)
explainer = shap.Explainer(predict_encoded, background_encoded.values)


# A first Pydantic Model
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal['Male', 'Female']
    country: str
    academic_level: Literal['Undergraduate', 'Graduate', 'High School']
    most_used_platform: Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter', 'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use: Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal['Medium', 'Low', 'Very High', 'High']


# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float
    # 6.777777 -> float


class ExplanationResponse(BaseModel):
    predicted_mental_health_score: float
    top_factors: list[dict]
    explanation: str


def build_input_row(data: StudentData) -> pd.DataFrame:
    country_group = data.country if data.country in top_countries else "Other"
    return pd.DataFrame([{
        'Age': data.age,
        'Gender': data.gender,
        'Country': data.country,
        'Academic_Level': data.academic_level,
        'Most_Used_Platform': data.most_used_platform,
        'Purpose_Of_Use': data.purpose_of_use,
        'Avg_Daily_Usage_Hours': data.avg_daily_usage_hours,
        'Daily_Unlocks': data.daily_unlocks,
        'Study_Hours': data.study_hours,
        'Physical_Activity_Hours': data.physical_activity_hours,
        'Sleep_Hours_Per_Night': data.sleep_hours_per_night,
        'Stress_Level': data.stress_level,
        'Grouped_country': country_group
    }])


@app.get('/')
def greet():
    return {"message": "Welcome to Sheryians AI School Guys"}


@app.post('/predict', response_model=PredictionResponse)  # 6.77777
def predict(data: StudentData):
    input_row = build_input_row(data)
    prediction = model.predict(input_row)[0]  # 6.77
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))


@app.post('/explain', response_model=ExplanationResponse)
def explain(data: StudentData):
    input_row = build_input_row(data)

    # 1. Get the raw prediction
    prediction = float(model.predict(input_row)[0])

    # 2. Compute SHAP values for this single prediction.
    #    Encode categoricals to numbers first, since SHAP's masker
    #    can't do isclose() comparisons on strings.
    input_encoded = encode_row(input_row).values
    shap_values = explainer(input_encoded)

    # shap_values.values[0] -> array of per-feature contributions
    # shap_values.base_values[0] -> the "average" prediction before features are applied
    contributions = dict(zip(FEATURE_COLUMNS, shap_values.values[0]))

    # 3. Sort by absolute impact, take top 3
    top_factors_raw = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    top_factors = [
        {
            "feature": feature,
            "value": input_row[feature].iloc[0],  # original (human-readable) value, not the encoded one
            "impact": round(float(impact), 3),
            "direction": "increased" if impact > 0 else "decreased"
        }
        for feature, impact in top_factors_raw
    ]

    # 4. Turn the SHAP numbers into a plain-language explanation via Groq
    factors_text = "\n".join(
        f"- {f['feature']} = {f['value']} ({f['direction']} the score by {abs(f['impact']):.2f})"
        for f in top_factors
    )

    prompt = f"""A machine learning model predicted a student's mental health score as {round(prediction, 2)} out of 10 (higher is better).

The top contributing factors from SHAP analysis were:
{factors_text}

Write a warm, plain-language explanation (2-3 sentences, no jargon, no SHAP/technical terms) of why this score came out this way, followed by one concrete, actionable suggestion. Do not diagnose or use clinical language."""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # fast + free-tier friendly on Groq
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    explanation_text = completion.choices[0].message.content

    return ExplanationResponse(
        predicted_mental_health_score=round(prediction, 2),
        top_factors=top_factors,
        explanation=explanation_text
    )