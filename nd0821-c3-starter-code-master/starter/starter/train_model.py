# Script to train machine learning model.

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model


# Get the path to the census dataset
current_dir = os.getcwd()
data_path = os.path.join(current_dir, '../data/census.csv')
# Load in the dataset
data=pd.read_csv(data_path)

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
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label='salary', training=False
)

# Train and save a model.
model = train_model(X_train, y_train)
#Save the model in model folder
model_filename = 'random_forest_model.pkl'
model_path = os.path.join(current_dir, '../model', model_filename)
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save encoder in model folder
encoder_filename = 'encoder.pkl'
encoder_path = os.path.join(current_dir, '../model', encoder_filename)
with open(encoder_path, 'wb') as encoder_file:
    pickle.dump(encoder, encoder_file)
