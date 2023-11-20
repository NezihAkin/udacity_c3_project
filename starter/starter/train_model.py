# Script to train machine learning model.
import pandas as pd
from ml.data import process_data
from ml.model import (
    calculate_performance_slicing,
    compute_model_metrics,
    inference,
    save_model,
    train_model,
)
from sklearn.model_selection import train_test_split

# Get the path to the census dataset
# current_dir = os.getcwd()
# data_path = os.path.join(current_dir, '../data/census.csv')
# Load in the dataset
data = pd.read_csv("starter/data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)
# Save the model in model folder
model_filename = "random_forest_model.pkl"
save_model(model_filename, model)

# Save encoder in model folder
encoder_filename = "encoder.pkl"
save_model(encoder_filename, encoder)

# Save trained Label Binarizer
lb_filename = "lb.pkl"
save_model(lb_filename, lb)

# Calculate predictions for X_test
preds = inference(model, X_test)

# Calculate performance metrics for the model
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Beta: {fbeta}")

# Calculate sliced data performances for occupation
calculate_performance_slicing(data, "occupation", model, encoder, lb)
