import time
import pandas as pd
import joblib

DATA_PATH = "train.csv"
MODEL_PATH = "model.pkl"
DROP_COLUMNS = ["Survived", "PassengerId", "Name", "Ticket", "Cabin"]

def load_unseen():
    df = pd.read_csv(DATA_PATH)
    if "Survived" not in df.columns:
        raise ValueError("train.csv must contain the Survived column.")
    # These are held-out records from the provided labelled file; labels are not passed to the model.
    unseen = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    for col in ["Sex", "Embarked"]:
        if col in unseen.columns:
            unseen[col] = unseen[col].astype("object")
    return unseen.iloc[700:730].copy()

def main():
    model = joblib.load(MODEL_PATH)
    records = load_unseen()
    if len(records) < 30:
        raise ValueError("Expected at least 730 rows in train.csv for the inference examples.")

    online = records.iloc[[0]]
    online_pred = int(model.predict(online)[0])
    print("TASK 2 — ONLINE INFERENCE")
    print("One passenger prediction:", "Survived" if online_pred == 1 else "Did not survive", f"({online_pred})")

    batch = records.iloc[:20].copy()
    batch["Predicted_Survived"] = model.predict(batch).astype(int)
    batch["Prediction_Label"] = batch["Predicted_Survived"].map({0: "Did not survive", 1: "Survived"})
    batch.to_csv("batch_predictions.csv", index=False)
    print("\nTASK 3 — BATCH INFERENCE")
    print(batch[["Predicted_Survived", "Prediction_Label"]].to_string(index=False))
    print("Saved batch_predictions.csv with", len(batch), "predictions")

    print("\nTASK 4 — STREAMING INFERENCE")
    for i in range(20, 30):
        record = records.iloc[[i]]
        prediction = int(model.predict(record)[0])
        print(f"Incoming record {i-19}/10 -> prediction={prediction}: " +
              ("Survived" if prediction == 1 else "Did not survive"))
        time.sleep(1)

if __name__ == "__main__":
    main()
