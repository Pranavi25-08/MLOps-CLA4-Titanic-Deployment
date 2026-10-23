import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "train.csv"
MODEL_PATH = "model.pkl"

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("train.csv not found. Put Kaggle's Titanic train.csv in this folder.")
    df = pd.read_csv(DATA_PATH)
    if "Survived" not in df.columns:
        raise ValueError("Expected the Kaggle Titanic training file with a 'Survived' column.")

    # Keep the target separate; remove identifiers/text fields that are not useful features.
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    X = X.copy()
    for col in ["Sex", "Embarked"]:
        if col in X.columns:
            X[col] = X[col].astype("object")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Refit the same pipeline on all labelled records for the final reusable artifact.
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    metadata = {
        "accuracy": float(accuracy),
        "features": X.columns.tolist(),
        "model": "RandomForestClassifier",
        "target": "Survived"
    }
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, indent=2)
    print(f"Dataset rows: {len(df)}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print(f"Saved reusable pipeline: {MODEL_PATH}")
    print("Features:", X.columns.tolist())

if __name__ == "__main__":
    main()
