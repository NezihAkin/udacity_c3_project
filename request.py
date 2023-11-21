import requests

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

response = requests.post(
    "https://udacity-c3-project.onrender.com/inference",
    json=input_data
)

print(f'Status code: {response.status_code}')
print(f'Inference Result: {response.json()}')