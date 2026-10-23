# CLA 4 — Observations

## 1. How does Online Inference differ from Batch Inference?
Online inference predicts one record at a time when a request arrives and is useful when a quick response is needed. Batch inference processes many records together, making it efficient for scheduled or bulk predictions.

## 2. What is the difference between Streaming Inference and Batch Inference?
Streaming inference processes records continuously as they arrive, one by one or in small events, with low latency. Batch inference waits until a group of records is collected and processes that group together.

## 3. Why is FastAPI used for model serving?
FastAPI makes it straightforward to expose a trained model through HTTP endpoints. It supports request validation with Pydantic, automatically generated interactive API documentation at `/docs`, and convenient integration with Python ML pipelines.

## 4. Why should the same trained model be reused for all inference strategies?
Reusing the same saved model keeps predictions consistent across online, batch, streaming, and API use. It avoids differences caused by training separate models and makes deployment easier to reproduce.

## Model evaluation note
The reported accuracy is measured on a held-out split of the Kaggle `train.csv` data. The inference demonstration uses records whose labels are not supplied to the prediction function. Because the Kaggle training file contains labels for every row, these are unseen-to-the-inference-call records rather than a separate external dataset.
