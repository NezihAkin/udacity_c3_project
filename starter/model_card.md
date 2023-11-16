# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is trained on 16.11.2023.
This is a Random Forest Classifier model from scikit-learn which aims to classify whether or not an individual has an income of over $50.000 based on various demographic features. 
The model is trained on the UCI Census Income Dataset. 
For hyperparameter tuning purpose, GridSearchCV from scikit-learn is used. GridSearchCV trains the RandomForest Classifier by splitting the data into 5 folds and using one of them as validation dataset in each turn. This method is called as cross-validation. 
GridSearchCV combines cross-validation with grid search that trains the combinations of the given parameters and selects the best performing model parameters. Given parameters for hyperparameter tuning are as follows:

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 4]
    }

In the training phase, OneHotEncoder for categorical variables and LabelBinarizer for the target is used to create numerical variables.

## Intended Use

Intended use of the model is educational purposes. This model is developed as a part of a project 'Deploying a ML model to the Cloud Application PLatform with FastAPI' from Udacity - Machine Learning DevOps Engineer course.

## Training Data

The dataset used for training is UCI Census Income Dataset.
This dataset covers many demographical information of individuals with a label of salary that indicates whether or not the individual has an income over $50.000.
Dataset covers information for 32.561 individuals. For training, 80% of the dataset has been taken as training data. However, training is completed with cross-validation that selects 80% of the data randomly for k different times. This means that all of the data is used for the whole training process.

## Evaluation Data

20% of the dataset has been used as evaluation dataset. Since cross-validation method is implemented for training and test, the whole data points used in the evaluation dataset at least one time covers more than 20% of the whole dataset.

## Metrics

Metrics used for model performance:

Precision = True Positives/(True Positives + False Positives)

Recall = True Positives / (True Positives + False Negatives)

F-Beta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)

Performance of the model:

Precision: 0.78

Recall: 0.54

F-beta: 0.64

## Ethical Considerations

Even though dataset used for training of this model is totally annonymized and aggregated, usage of demographical and personal data should be approached with caution. All permissions to collect and use the dataset should be in place before the model development project start. UCI Census Income Dataset is a public dataset that is licensed and can be used for research purposes. For this model development, it was appropriate to use this dataset.

After the model development, fairness of the model should be checked. Because, dataset contains many demographical data such as age, place, gender, race and these features should be treated carefully to not cause any unintentional but harmful discrimination. Model fairness should be checked in that aspect.

## Caveats and Recommendations

This model development is only for research and educational purposes and it is not allowed to be used commercially.
