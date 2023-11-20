import os
import pickle

from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

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


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4],
    }

    # Create RandomForestClassifier instance
    rf_classifier = RandomForestClassifier(random_state=42)

    # Create GridSearchCV instance
    grid_search = GridSearchCV(
        estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )

    # Fit hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def save_model(filename, file):
    """Saves model or encoders as a pickle file to model folder.

    Inputs
    ------
    filename : str
        Name of the file.
    file : pickle
        file to be saved as pickle (either model or encoder)
    """

    path = os.path.join("../model", filename)
    with open(path, "wb") as save_file:
        pickle.dump(file, save_file)


def calculate_performance_slicing(df, feature, model, encoder, lb):
    """Calculates performance metrics on slices of categorical features.

    Inputs
    ------
    df: pd.DataFrame
        Training data
    feature : str
        List of the names of categorical features
    model : RandomForestClassifier
        Trained model
    encoder : OneHotEncoder
        Encoder for categorical variables
    lb : LabelBinarizer
        Label Binarizer
    """

    print(f"Feature: {feature}", "\n")
    with open(f"../screenshots/slice_output_{feature}.txt", "w") as f:
        for value in df[feature].unique():
            df_temp = df[df[feature] == value]
            _, test = train_test_split(df_temp, test_size=0.20)

            X_test, y_test, encoder, lb = process_data(
                test,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Calculate predictions for X_test
            preds = inference(model, X_test)

            # Calculate performance metrics for the model
            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            print(f"Precision for {value}: {precision}")
            print(f"Recall for {value}: {recall}")
            print(f"Fbeta for {value}: {fbeta}")

            # save to a txt file
            lines = [
                f"Feature: {feature}",
                "\n",
                f"Precision for {value}: {precision}",
                "\n" f"Recall for {value}: {recall}",
                "\n" f"Fbeta for {value}: {fbeta}",
            ]

            f.writelines(lines)
