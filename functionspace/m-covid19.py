# covid19_pipeline.py

import pandas as pd
import pickle
from prophet import Prophet
import plotly.graph_objects as go
import os

def fetch_covid19_data(
    url: str = "https://covid.ourworldindata.org/data/owid-covid-data.csv",
    output_file: str = "dataspace/covid19_data.csv",
    country: str = "United States"
) -> pd.DataFrame:
    """
    Fetches COVID-19 data for a specific country and saves it to a CSV file.

    Parameters:
        url (str): URL to fetch the COVID-19 data CSV.
        output_file (str): Path to save the filtered COVID-19 data.
        country (str): Country name to filter the data.

    Returns:
        pd.DataFrame: DataFrame containing COVID-19 data for the specified country.
    """
    df = pd.read_csv(url)
    df_country = df[df['location'] == country]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_country.to_csv(output_file, index=False)
    return df_country

def preprocess_covid19_data(
    input_file: str = "dataspace/covid19_data.csv",
    output_file: str = "dataspace/covid19_preprocessed.csv",
    date_column: str = "date",
    target_column: str = "new_cases_smoothed",
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    Preprocesses COVID-19 data for time-series forecasting.

    Parameters:
        input_file (str): Path to the CSV file containing COVID-19 data.
        output_file (str): Path to save the preprocessed data.
        date_column (str): Name of the column containing dates.
        target_column (str): Name of the column to forecast.
        fill_missing (bool): Whether to fill missing values.

    Returns:
        pd.DataFrame: DataFrame ready for time-series modeling.
    """
    df = pd.read_csv(input_file)
    df = df[[date_column, target_column]]
    if fill_missing:
        df[target_column].fillna(method='ffill', inplace=True)
        df[target_column].fillna(method='bfill', inplace=True)
    df = df.dropna()
    df.columns = ['ds', 'y']  # Prophet requires columns 'ds' and 'y'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    return df

def build_and_train_prophet_model(
    input_file: str = "dataspace/covid19_preprocessed.csv",
    model_save_path: str = "dataspace/prophet_model.pkl"
) -> Prophet:
    """
    Builds and trains a Prophet model on the preprocessed data.

    Parameters:
        input_file (str): Path to the preprocessed CSV file.
        model_save_path (str): Path to save the trained model.

    Returns:
        Prophet: Trained Prophet model.
    """
    df = pd.read_csv(input_file)
    model = Prophet()
    model.fit(df)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # Save the model to a file
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    return model

def forecast_covid19_cases(
    periods: int = 30,
    model_load_path: str = "dataspace/prophet_model.pkl",
    forecast_save_path: str = "dataspace/covid19_forecast.csv"
) -> pd.DataFrame:
    """
    Uses the trained Prophet model to forecast future COVID-19 cases.

    Parameters:
        periods (int): Number of days to forecast into the future.
        model_load_path (str): Path to load the trained model.
        forecast_save_path (str): Path to save the forecasted data.

    Returns:
        pd.DataFrame: DataFrame containing forecasted values.
    """
    # Load the model
    with open(model_load_path, 'rb') as f:
        model = pickle.load(f)
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    os.makedirs(os.path.dirname(forecast_save_path), exist_ok=True)
    forecast.to_csv(forecast_save_path, index=False)
    return forecast


import pandas as pd
import plotly.graph_objects as go
import os

def visualize_forecast(
    forecast_file: str = "dataspace/covid19_forecast.csv",
    actuals_file: str = "dataspace/covid19_preprocessed.csv",
    output_html_file: str = "dataspace/covid19_forecast.html"
) -> str:
    """
    Creates an interactive plot of the COVID-19 forecast alongside actual data.

    Parameters:
        forecast_file (str): Path to the CSV file containing forecasted data.
        actuals_file (str): Path to the CSV file containing actual data.
        output_html_file (str): Path to save the forecast plot as an HTML file.

    Returns:
        str: Confirmation message indicating the plot has been saved.
    """
    forecast = pd.read_csv(forecast_file)
    actuals = pd.read_csv(actuals_file)
    
    fig = go.Figure()
    # Add actual cases
    fig.add_trace(go.Scatter(
        x=actuals['ds'], y=actuals['y'], mode='markers', name='Actual Cases',
        marker=dict(color='blue')
    ))
    # Add forecasted cases
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast',
        line=dict(color='red')
    ))
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Confidence Interval',
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Confidence Interval',
        fill='tonexty', fillcolor='rgba(68, 68, 68, 0.1)', line=dict(width=0), showlegend=False
    ))
    fig.update_layout(
        title='COVID-19 Cases Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Cases',
        legend_title='Legend',
    )
    os.makedirs(os.path.dirname(output_html_file), exist_ok=True)
    fig.write_html(output_html_file)
    print(f"Forecast visualization saved to {output_html_file}")
    return "COVID-19 forecast plot saved."



# # Example usage
# if __name__ == "__main__":
#     # Step 1: Fetch data
#     df_covid = fetch_covid19_data(country="United States")

#     # Step 2: Preprocess data
#     df_preprocessed = preprocess_covid19_data()

#     # Step 3: Build and train the model
#     prophet_model = build_and_train_prophet_model()

#     # Step 4: Forecast future cases
#     forecast_df = forecast_covid19_cases(periods=30)

#     # Step 5: Visualize the forecast
#     message = visualize_forecast()
#     print(message)
