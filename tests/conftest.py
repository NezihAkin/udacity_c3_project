import pytest
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="session")
def data():
    df = pd.DataFrame(np.random.randint(0, 500, size=(300, 5), ), columns=["A", "B", "C", "D", "E"])
    df["y_true"] = np.random.randint(2, size=(300,))
    y = df.pop("y_true")
    X = df

    return X, y


@pytest.fixture(scope="session")
def model(data):
    X, y = data
    model = RandomForestClassifier(random_state=23)
    model.fit(X, y)

    return model

@pytest.fixture(scope="session")
def pred_label_1():
    input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
}
    return input_data

@pytest.fixture(scope="session")
def pred_label_2():
    input_data = {
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "India"
}
    return input_data