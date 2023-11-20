from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import compute_model_metrics, inference, train_model

def test_train_model(data):
    X, y = data
    model = train_model(X,y)
    assert isinstance(model, RandomForestClassifier)

def test_inference(model, data):
    X_test, _ = data
    preds = inference(model, X_test)

    assert preds.size > 0
    assert ((preds >= 0) & (preds <= 1)).all()


def test_metrics(data, model):
    X, y = data
    preds = inference(model,X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision >= 0 and precision <= 1
    assert recall >= 0 and recall <= 1
    assert fbeta >= 0 and fbeta <= 1