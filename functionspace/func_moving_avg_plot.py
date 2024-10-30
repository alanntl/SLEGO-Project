import ast
import pandas as pd
import datetime
from yfinance import Ticker as si
from typing import Union, Callable
import plotly.graph_objects as go
from typing import Union

'''
Guidelines for building microservices (python functions):

1. Create a python function
2. Make sure the function has a docstring that explain to user how to use the microservice, and what the service does
3. Make sure the function has a return statement
4. Make sure the function has a parameter
5. Make sure the function has a default value for the parameter
6. Make sure the function has a type hint for the parameter 

'''
def import_marketdata_yahoo_csv(ticker: str = 'msft', 
                                start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
                                end_date: str = datetime.datetime.now().strftime("%Y-%m-%d"), 
                                output_file_path: str = 'dataspace/dataset.csv' ):
    """
    Imports market data from Yahoo using the yfinance Ticker API.
    
    Parameters:
    - ticker (str): The stock ticker symbol. Default is 'msft'.
    - start_date (str): The start date for the data in 'YYYY-MM-DD' format. Default is one year ago from today.
    - end_date (str): The end date for the data in 'YYYY-MM-DD' format. Default is today.
    - output_file_path (str): The path to save the resulting CSV file. Default is 'yfinance_ohlc.csv'.
    
    Returns:
    - pd.DataFrame: A dataframe containing the imported market data.
    """

    # Fetch the market data using the yfinance Ticker
    df = si(ticker).history(start=start_date, end=end_date)
    df['Datetime'] = df.index
    df.to_csv(output_file_path)
    # Return the StringIO object
    return df

def preprocess_filling_missing_values(
    input_file_path: str = 'dataspace/dataset.csv',
    output_file_path: str = 'dataspace/dataset.csv',
    fill_strategy: Union[str, float, int, dict, Callable] = 'ffill',
):
    """
    Preprocesses a CSV dataset by filling missing values using various strategies and outputs the processed dataset to a CSV file.

    Parameters:
    - input_file_path (str): Path to the input CSV dataset. Defaults to 'dataspace/dataset.csv'.
    - output_file_path (str): Path for the output CSV dataset where processed data will be saved. Defaults to 'dataspace/dataset.csv'.
    - fill_strategy (Union[str, float, int, dict, Callable]): Defines the strategy for filling missing values. Can be a string ('ffill' for forward fill, 'bfill' for backward fill), a scalar value to fill missing entries, a dictionary to specify method or value per column, or a callable with custom logic. Defaults to 'ffill'.

    Returns:
    pandas.DataFrame: The processed DataFrame with missing values filled.

    This function reads a CSV file, fills missing values according to the specified strategy, and writes the processed DataFrame to the specified CSV file. It supports various filling strategies to accommodate different data preprocessing needs.
    """

    data = pd.read_csv(input_file_path)

    if callable(fill_strategy):
        # Apply a custom function to fill missing values
        data.apply(lambda x: x.fillna(fill_strategy(x)), axis=0)
    else:
        # Apply predefined pandas strategies or scalar values
        data.fillna(value=fill_strategy, inplace=True)

    data.to_csv(output_file_path)

    return data


def plot_chart_local(input_file_path: str = 'dataspace/dataset.csv', 
                     index_col: Union[None, int, str] = 0,
                     x_column: str = 'Date', 
                     y_column: str = 'SMA_Close', 
                     title: str = 'Data Plot', 
                     legend_title:str= 'Legend',
                     mode:str = 'lines',
                     output_html_file_path: str = 'dataspace/dataset_plot.html'):
    """
    Create a Plotly graph for data and save it as an HTML file locally.
    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset.csv'.
    :param x_column: Column to use for the X-axis. If None, the data index is used. Default is 'Date'.
    :param y_column: Column to use for the Y-axis. Default is 'SMA_Close'.
    :param title: Title of the plot. Default is 'Data Plot'.
    :param output_html_file_path: Local file path where the HTML file will be saved. Default is 'dataspace/dataset_plot.html'.
    """

    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=index_col)

    # Determine X-axis data
    x_data = data[x_column] if x_column else data.index

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=data[y_column], mode=mode, name=y_column))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=legend_title
    )

    # Save the figure as an HTML file locally
    fig.write_html(output_html_file_path)

    return "Plot saved locally."

def compute_simple_moving_average_local(input_file_path: str = 'dataspace/dataset.csv', 
                                        column_name: str = 'Close', 
                                        window_size: int = 20, 
                                        output_file_path: str = 'dataspace/dataset.csv'):
    """
    Compute simple moving average for a specified column in the data and save updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset.csv'.
    :param column_name: Name of the column to calculate the SMA on. Default is 'Close'.
    :param window_size: Window size for the moving average. Default is 20.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_sma.csv'.
    """
    # Load data from a local file
    
    data = pd.read_csv(input_file_path, index_col=0)

    # Calculate Simple Moving Average
    if column_name in data.columns:
        data[f'SMA_{column_name}'] = data[column_name].rolling(window=window_size).mean()
    else:
        raise ValueError(f"Column '{column_name}' not found in data")

    # Save updated data to a local file
    data.to_csv(output_file_path)

    return data



