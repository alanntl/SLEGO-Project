# regression_pipeline.py

import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Union

def fetch_sklearn_dataset(
    dataset_name: str = 'california_housing',
    output_csv_file: str = 'dataspace/housing_dataset.csv'
):
    """
    Fetches a dataset from scikit-learn's datasets and saves it as a CSV file.

    Parameters:
    - dataset_name (str): Name of the dataset to fetch ('california_housing').
    - output_csv_file (str): Path to save the dataset as a CSV file.

    Returns:
    - pd.DataFrame: The dataset as a pandas DataFrame.
    """
    if dataset_name == 'california_housing':
        data = fetch_california_housing(as_frame=True)
    else:
        raise ValueError("Unsupported dataset_name. Choose 'california_housing'.")

    df = pd.concat([data.data, data.target.rename('target')], axis=1)
    df.to_csv(output_csv_file, index=False)
    return df

def train_sklearn_regression_model(
    input_file_path: str = 'dataspace/housing_dataset.csv',
    target_column: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
    model_output_path: str = 'dataspace/sklearn_regression_model.pkl',
    performance_output_path: str = 'dataspace/sklearn_regression_performance.txt'
):
    """
    Trains a Linear Regression model and saves it.

    Parameters:
    - input_file_path (str): Path to the dataset CSV file.
    - target_column (str): Name of the target column.
    - test_size (float): Fraction of data to use as test set.
    - random_state (int): Seed for reproducibility.
    - model_output_path (str): Path to save the trained model.
    - performance_output_path (str): Path to save performance metrics.

    Returns:
    - dict: Dictionary containing performance metrics.
    """
    df = pd.read_csv(input_file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save the model
    joblib.dump(model, model_output_path)

    # Save performance metrics
    with open(performance_output_path, 'w') as f:
        f.write(f'MSE: {mse}\n')
        f.write(f'R2 Score: {r2}\n')

    return {'MSE': mse, 'R2 Score': r2}

def sklearn_model_predict(
    model_path: str = 'dataspace/sklearn_regression_model.pkl',
    input_data_path: str = 'dataspace/housing_test.csv',
    output_data_path: str = 'dataspace/housing_predictions.csv'
):
    """
    Loads a trained sklearn model and makes predictions on new data.

    Parameters:
    - model_path (str): Path to the saved sklearn model.
    - input_data_path (str): Path to the CSV file containing new data.
    - output_data_path (str): Path to save the predictions.

    Returns:
    - pd.Series: Predictions made by the model.
    """
    model = joblib.load(model_path)
    X_new = pd.read_csv(input_data_path)
    predictions = model.predict(X_new)

    # Save predictions to CSV
    output_df = X_new.copy()
    output_df['Predicted_Target'] = predictions
    output_df.to_csv(output_data_path, index=False)

    return predictions
