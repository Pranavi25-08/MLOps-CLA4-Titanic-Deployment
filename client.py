import requests

BASE_URL = "http://127.0.0.1:8000"

payload = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
print("Status:", response.status_code)
print("Response:", response.json())
response.raise_for_status()
