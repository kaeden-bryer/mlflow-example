import os
import warnings
import sys
import time

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    medae = median_absolute_error(actual, pred)
    return rmse, mae, r2, mape, medae


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets
    train, test = train_test_split(data, random_state=42)

    # The predicted column is "quality"
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    start_time = time.time()

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    training_time = time.time() - start_time

    predicted_qualities = lr.predict(test_x)

    rmse, mae, r2, mape, medae = eval_metrics(test_y, predicted_qualities)

    print(f"ElasticNet model (alpha={alpha:.3f}, l1_ratio={l1_ratio:.3f})")
    print(f"  RMSE: {rmse}")
    print(f"  MAE: {mae}")
    print(f"  R2: {r2}")
    print(f"  MAPE: {mape}")
    print(f"  Median AE: {medae}")
    print(f"  Training time (s): {training_time}")

    # Tags
    mlflow.set_tag("model_type", "ElasticNet")
    mlflow.set_tag("dataset", "wine-quality")
    mlflow.set_tag("course_lab", "mlflow_tracking")

    # Parameters
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_param("n_train_samples", len(train_x))
    mlflow.log_param("n_test_samples", len(test_x))
    mlflow.log_param("n_features", train_x.shape[1])
    mlflow.log_param("train_test_ratio", round(len(train_x) / len(test_x), 4))

    # Metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("median_absolute_error", medae)
    mlflow.log_metric("training_time_seconds", training_time)

    # Model statistics
    intercept_value = float(np.ravel(lr.intercept_)[0]) if np.ndim(lr.intercept_) > 0 else float(lr.intercept_)
    mlflow.log_metric("intercept", intercept_value)
    mlflow.log_metric("coef_mean", float(np.mean(lr.coef_)))
    mlflow.log_metric("coef_std", float(np.std(lr.coef_)))
    mlflow.log_metric("coef_min", float(np.min(lr.coef_)))
    mlflow.log_metric("coef_max", float(np.max(lr.coef_)))

    # Save coefficients as artifact
    coef_df = pd.DataFrame({
        "feature": train_x.columns,
        "coefficient": lr.coef_
    })
    coef_df.to_csv("model_coefficients.csv", index=False)
    mlflow.log_artifact("model_coefficients.csv")

    # Log model
    mlflow.sklearn.log_model(lr, "model")
