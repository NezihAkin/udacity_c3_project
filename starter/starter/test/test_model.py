import pytest
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, train_model, inference

@pytest.fixture
def data():
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir,'starter/data/census.csv')
    df = pd.read_csv(data_path)
    return df

@pytest.fixture
def cat_features():
    return ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

def test_process_data(data):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=False
    )

    assert not X_train.empty
    assert len(y_train)>0
    assert not X_test.empty
    assert len(y_test)>0

def test_inference(data, cat_features):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)

    preds = inference(y_train, model)

    assert len(preds)>0
    assert preds.between(0,1)

def test_metrics():
    rng = np.random.default_rng(42)
    y = rng.random(10)
    preds = rng.random(10)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision>=0 and precision<=1
    assert recall>=0 and recall<=1
    assert fbeta>=0 and fbeta<=1