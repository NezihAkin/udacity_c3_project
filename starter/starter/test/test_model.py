import pytest
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, train_model, inference

@pytest.fixture
def data():
    df = pd.read_csv('starter/data/census.csv')
    return df

@pytest.fixture
def cat_features():
    return ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

def test_process_data(data, cat_features):
    train, _ = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    assert X_train.size>0
    assert y_train.size>0

def test_inference(data, cat_features):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)

    preds = inference(model, X_train)

    assert preds.size>0
    assert ((preds>=0)&(preds<=1)).all()

def test_metrics():
    rng = np.random.default_rng(42)
    y = rng.integers(0,1,10)
    preds = rng.integers(0,1,10)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision>=0 and precision<=1
    assert recall>=0 and recall<=1
    assert fbeta>=0 and fbeta<=1