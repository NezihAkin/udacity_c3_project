import pytest
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, train_model, inference

@pytest.fixture(scope='session')
def data():
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir,'../data/census.csv')
    df = pd.read_csv(data_path)
    return df

@pytest.fixture(scope='session')
def process_data(data):
    train, test = train_test_split(data, test_size=0.20)
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
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=False
    )
    return X_train, X_test, y_train, y_test

@pytest.fixture(scope='session')
def train_model_and_predict(X_train, y_train):
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    return model, preds

def test_process_data(X_train, X_test, y_train, y_test):
    assert not X_train.empty
    assert len(y_train)>0
    assert not X_test.empty
    assert len(y_test)>0

def test_inference(X_train, model):
    preds = inference(X_train, model)
    assert len(preds)>0
    assert preds.between(0,1)

def test_metrics(y_train, preds):
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    assert precision>=0 and precision<=1
    assert recall>=0 and recall<=1
    assert fbeta>=0 and fbeta<=1