import json
import pickle
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load the trained model and encoder
model_filepath = "model/random_forest_model.pkl"
encoder_filepath = "model/encoder.pkl"

with open(model_filepath, "rb") as file:
    model = pickle.load(file)

with open(encoder_filepath, "rb") as file:
    encoder = pickle.load(file)

# Instantiate the app
app = FastAPI()


class Data(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True


# GET on greeting
@app.get("/")
async def welcome_intent():
    return {"greeting": "Welcome to my Model!"}


# POST for model inference
@app.post("/inference")
async def inference(data: Data):
    # Converting input data into Pandas DataFrame
    input_df = pd.json_normalize([jsonable_encoder(data)])

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=None
    )

    # Getting the prediction from the Random Forest Model
    pred = model.predict(X)
    if pred.tolist()[0] == 0:
        pred_label = "<=50K"
    else:
        pred_label = ">50K"

    return {"Prediction": pred_label}