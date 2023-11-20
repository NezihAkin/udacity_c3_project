from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_say_hello():
    request = client.get("/")
    assert request.status_code == 200
    assert request.json() == {"greeting":"Welcome to my Model!"}

def test_inference_1(pred_label_1):
    request = client.post("/inference/", json=pred_label_1)
    assert request.status_code == 200
    assert request.json() == {"Prediction": "<=50K"}

def test_inference_2(pred_label_2):
    request = client.post("/inference/", json=pred_label_2)
    assert request.status_code == 200
    assert request.json() == {"Prediction": ">50K"}