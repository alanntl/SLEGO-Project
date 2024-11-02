# environmental_anomaly_detection_pipeline.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import requests
import os

def fetch_air_quality_data(
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
    extract_to: str = "dataspace/",
    output_csv_file: str = "dataspace/AirQuality.csv"
):
    """
    Fetches the Air Quality dataset from the UCI repository, extracts it, and saves it as a CSV file.

    Parameters:
    - url (str): URL to download the dataset ZIP file.
    - extract_to (str): Directory to extract the ZIP contents.
    - output_csv_file (str): Path to save the extracted CSV file.

    Returns:
    - pd.DataFrame: The Air Quality dataset as a pandas DataFrame.
    """
    import zipfile
    import io

    # Ensure the dataspace directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extract the CSV file
            z.extract("AirQualityUCI.csv", path=extract_to)
        
        # Read the CSV file
        df = pd.read_csv(os.path.join(extract_to, "AirQualityUCI.csv"), sep=';', decimal=',')
        
        # Drop the last two columns which are empty
        df = df.iloc[:, :-2]
        
        # Replace comma with dot and convert to numeric
        df = df.replace(',', '.', regex=True)
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save the cleaned dataset
        df.to_csv(output_csv_file, index=False)
        return df
    else:
        raise ConnectionError(f"Failed to download dataset. Status code: {response.status_code}")

def preprocess_air_quality_data(
    input_file_path: str = "dataspace/AirQuality.csv",
    output_file_path: str = "dataspace/preprocessed_AirQuality.csv",
    target_column: str = "NO2(GT)"
):
    """
    Preprocesses the Air Quality dataset by handling missing values and selecting relevant features.

    Parameters:
    - input_file_path (str): Path to the raw CSV dataset.
    - output_file_path (str): Path to save the preprocessed CSV dataset.
    - target_column (str): The column to analyze for anomalies.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    df = pd.read_csv(input_file_path)
    
    # Convert 'Date' and 'Time' to datetime
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    
    # Drop rows with invalid datetime
    df.dropna(subset=['Datetime'], inplace=True)
    
    # Set datetime as index
    df.set_index('Datetime', inplace=True)
    
    # Select relevant columns (example: CO, NO2, SO2, etc.)
    relevant_columns = [
        "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)",
        "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"
    ]
    df = df[relevant_columns]
    
    # Handle missing values by forward filling
    df.fillna(method='ffill', inplace=True)
    
    # Save the preprocessed data
    df.to_csv(output_file_path)
    return df

def detect_anomalies_isolation_forest(
    input_file_path: str = "dataspace/preprocessed_AirQuality.csv",
    target_column: str = "NO2(GT)",
    output_file_path: str = "dataspace/air_quality_anomalies.csv"
):
    """
    Detects anomalies in the specified target column using Isolation Forest.

    Parameters:
    - input_file_path (str): Path to the preprocessed CSV dataset.
    - target_column (str): Name of the column to analyze for anomalies.
    - output_file_path (str): Path to save the results with anomaly labels.

    Returns:
    - pd.DataFrame: DataFrame with anomaly labels.
    """
    df = pd.read_csv(input_file_path, parse_dates=['Datetime'], index_col='Datetime')
    
    # Select the target column
    data = df[[target_column]].dropna()
    
    # Initialize Isolation Forest
    model = IsolationForest(contamination=0.01, random_state=42)
    data['Anomaly'] = model.fit_predict(data)
    
    # Convert prediction to boolean
    data['Anomaly'] = data['Anomaly'].apply(lambda x: True if x == -1 else False)
    
    # Merge anomaly labels back to the original dataframe
    df = df.join(data['Anomaly'], how='left')
    df['Anomaly'].fillna(False, inplace=True)
    
    # Save the results
    df.to_csv(output_file_path)
    return df

def plot_anomalies(
    input_file_path: str = "dataspace/air_quality_anomalies.csv",
    target_column: str = "NO2(GT)",
    date_column: str = "Datetime",
    anomaly_column: str = "Anomaly",
    output_html_file: str = "dataspace/air_quality_anomaly_plot.html"
):
    """
    Plots the time series data and highlights the detected anomalies.

    Parameters:
    - input_file_path (str): Path to the CSV file containing data and anomaly labels.
    - target_column (str): Name of the column to plot.
    - date_column (str): Name of the date column.
    - anomaly_column (str): Name of the anomaly label column.
    - output_html_file (str): Path to save the interactive Plotly HTML plot.

    Returns:
    - None: Saves the plot to the specified HTML file.
    """
    df = pd.read_csv(input_file_path, parse_dates=[date_column])
    
    # Create the main time series plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df[date_column],
            y=df[target_column],
            mode='lines',
            name=target_column
        )
    )
    
    # Highlight anomalies
    anomalies = df[df[anomaly_column]]
    fig.add_trace(
        go.Scatter(
            x=anomalies[date_column],
            y=anomalies[target_column],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='circle-open')
        )
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title=f'Time Series Anomaly Detection for {target_column}',
        xaxis_title='Datetime',
        yaxis_title=target_column,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    # Save the plot as an HTML file
    fig.write_html(output_html_file)
    print(f"Anomaly plot saved to {output_html_file}")