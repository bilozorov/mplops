import os
import pickle

import mlflow

if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_model_to_mlflow(data, *args, **kwargs):
    lr, dv = data
    os.makedirs("models", exist_ok=True)
    with open("models/dict_vectorizer.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    experiment_name = "homework3_exp1"
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "artifacts")
        mlflow.log_artifact("models/dict_vectorizer.b", artifact_path="preprocessor")