import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import vectorbt as vbt
# caclulate moviang average of stock price

def movingAvg(input_file_path: str = 'dataspace/dataset.csv', 
                                        column_name: str = 'Close', 
                                        window_size: int = 50, 
                                        output_file_path: str = 'dataspace/dataset_avg50.csv'):
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

def movingAvg_50(input_file_path: str = 'dataspace/dataset.csv', 
                                        column_name: str = 'Close', 
                                        window_size: int = 50, 
                                        output_file_path: str = 'dataspace/dataset_avg50.csv'):
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

def movingAvg_10(input_file_path: str = 'dataspace/dataset.csv', 
                                        column_name: str = 'Close', 
                                        window_size: int = 10, 
                                        output_file_path: str = 'dataspace/dataset_avg10.csv'):
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



def calculate_vwap(input_file_path: str = 'dataspace/dataset.csv', 
                   price_column: str = 'Close', 
                   volume_column: str = 'Volume', 
                   output_file_path: str = 'dataspace/dataset_vwap.csv'):
    """
    Calculate the Volume Weighted Average Price (VWAP) and save the result to a CSV file.
    
    Parameters:
    input_file_path (str): Path to the input CSV file.
    price_column (str): Name of the column containing the price data.
    volume_column (str): Name of the column containing the volume data.
    output_file_path (str): Path to the output CSV file where the data with VWAP will be saved.
    
    Returns:
    pd.DataFrame: DataFrame with the VWAP column added.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)

    # Check if the required columns are in the DataFrame
    if price_column not in data.columns or volume_column not in data.columns:
        raise ValueError(f"DataFrame must contain '{price_column}' and '{volume_column}' columns")

    # Calculate VWAP
    data['VWAP'] = (data[price_column] * data[volume_column]).cumsum() / data[volume_column].cumsum()

    # Save updated data to a local file
    data.to_csv(output_file_path)

    return data


def generate_mAvg_Crossover_trading_signals(price_file_path: str = 'dataspace/dataset.csv',
                                                               price_column: str = 'Close',
                                                               short_ma_file_path: str = 'dataspace/dataset_avg10.csv',
                                                               long_ma_file_path: str = 'dataspace/dataset_avg50.csv',
                                                               output_file_path: str = 'dataspace/dataset_signals.csv'):
    """
    Generate trading signals based on moving average crossover strategy using pre-calculated moving averages
    and save the result to a CSV file.

    Parameters:
    price_file (str): Path to the CSV file containing the price data.
    short_ma_file (str): Path to the CSV file containing the short-term moving average.
    long_ma_file (str): Path to the CSV file containing the long-term moving average.
    price_column (str): Name of the column containing the price data.
    output_file_path (str): Path to the output CSV file where the data with trading signals will be saved.

    Returns:
    pd.DataFrame: DataFrame with the trading signals added.
    """
    # Load data from the price and moving average files
    price_data = pd.read_csv(price_file_path, index_col=0)
    short_ma_data = pd.read_csv(short_ma_file_path, index_col=0)
    long_ma_data = pd.read_csv(long_ma_file_path, index_col=0)

    # Check if the price column is in the price DataFrame
    if price_column not in price_data.columns:
        raise ValueError(f"The price DataFrame must contain '{price_column}' column")

    # Merge data on the index (date)
    data = price_data[[price_column]].join(short_ma_data[['SMA_Close']].rename(columns={'SMA_Close': 'SMA_Short'}))
    data = data.join(long_ma_data[['SMA_Close']].rename(columns={'SMA_Close': 'SMA_Long'}))

    # Generate trading signals
    data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, 0)

    # Generate trading positions
    data['Position'] = data['Signal'].diff()

    # Save updated data to a local file
    data.to_csv(output_file_path)

    return data


def calculate_typical_price(input_file_path: str = 'dataspace/dataset.csv',
                            output_file_path: str = 'dataspace/dataset_typical_price.csv'):
    """
    Calculate the Typical Price for each period and save the updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset.csv'.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_typical_price.csv'.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)
    
    # Calculate Typical Price
    if {'High', 'Low', 'Close'}.issubset(data.columns):
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    else:
        raise ValueError("Data must contain 'High', 'Low', and 'Close' columns")

    # Save updated data to a local file
    data.to_csv(output_file_path)
    
    return data

def calculate_raw_money_flow_RMF(input_file_path: str = 'dataspace/dataset_typical_price.csv',
                             output_file_path: str = 'dataspace/dataset_raw_money_flow.csv'):
    """
    Calculate the Raw Money Flow for each period and save the updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset_typical_price.csv'.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_raw_money_flow.csv'.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)
    
    # Calculate Raw Money Flow
    if 'Typical_Price' in data.columns and 'Volume' in data.columns:
        data['Raw_Money_Flow'] = data['Typical_Price'] * data['Volume']
    else:
        raise ValueError("Data must contain 'Typical_Price' and 'Volume' columns")
    
    # Save updated data to a local file
    data.to_csv(output_file_path)
    
    return data

def calculate_money_flow_ratio_MFR(input_file_path: str = 'dataspace/dataset_raw_money_flow.csv',
                         period: int = 14,
                         output_file_path: str = 'dataspace/dataset_money_flow_ratio.csv'):
    """
    Calculate the Positive and Negative Money Flow for each period and save the updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset_raw_money_flow.csv'.
    :param period: The period for calculating the rolling sums. Default is 14.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_money_flow.csv'.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)
    
    # Calculate Positive and Negative Money Flow
    data['Positive_Flow'] = 0
    data['Negative_Flow'] = 0
    
    for i in range(1, len(data)):
        if data['Typical_Price'].iloc[i] > data['Typical_Price'].iloc[i - 1]:
            data.at[data.index[i], 'Positive_Flow'] = data['Raw_Money_Flow'].iloc[i]
        elif data['Typical_Price'].iloc[i] < data['Typical_Price'].iloc[i - 1]:
            data.at[data.index[i], 'Negative_Flow'] = data['Raw_Money_Flow'].iloc[i]
    
    data['Positive_Flow'] = data['Positive_Flow'].rolling(window=period).sum()
    data['Negative_Flow'] = data['Negative_Flow'].rolling(window=period).sum()
    
    # Save updated data to a local file
    data.to_csv(output_file_path)
    
    return data

def calculate_money_flow_index_MFI(input_file_path: str = 'dataspace/dataset_money_flow_ratio.csv',
                  output_file_path: str = 'dataspace/dataset_mfi.csv'):
    """
    Calculate the Money Flow Index (MFI) and save the updated data locally.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset_money_flow.csv'.
    :param period: The period for calculating the MFI. Default is 14.
    :param output_file_path: Local file path where the updated data will be saved. Default is 'dataspace/dataset_mfi.csv'.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)
    
    # Calculate Money Flow Ratio and MFI
    data['Money_Flow_Ratio'] = data['Positive_Flow'] / data['Negative_Flow']
    data['MFI'] = 100 - (100 / (1 + data['Money_Flow_Ratio']))
    
    # Save updated data to a local file
    data.to_csv(output_file_path)
    
    return data


def generate_mfi_trading_signals(input_file_path: str = 'dataspace/dataset_mfi.csv',
                                 mfi_column: str = 'MFI',
                                 threshold_overbought: int = 80,
                                 threshold_oversold: int = 20,
                                 output_file_path: str = 'dataspace/dataset_mfi_signals.csv'):
    """
    Generate trading signals based on the Money Flow Index (MFI) and save the result to a CSV file.

    :param input_file_path: Local file path for the input data file. Default is 'dataspace/dataset_mfi.csv'.
    :param mfi_column: Name of the column containing the MFI data. Default is 'MFI'.
    :param threshold_overbought: Threshold for overbought condition. Default is 80.
    :param threshold_oversold: Threshold for oversold condition. Default is 20.
    :param output_file_path: Local file path where the data with trading signals will be saved. Default is 'dataspace/dataset_mfi_signals.csv'.
    :return: pd.DataFrame with the trading signals added.
    """
    # Load data from a local file
    data = pd.read_csv(input_file_path, index_col=0)
    
    # Check if the MFI column is in the DataFrame
    if mfi_column not in data.columns:
        raise ValueError(f"DataFrame must contain '{mfi_column}' column")
    
    # Generate trading signals
    data['Signal'] = 0
    data.loc[data[mfi_column] > threshold_overbought, 'Signal'] = -1  # Sell signal
    data.loc[data[mfi_column] < threshold_oversold, 'Signal'] = 1     # Buy signal

    # Generate trading positions
    data['Position'] = data['Signal'].diff()
    
    # Save updated data to a local file
    data.to_csv(output_file_path)
    
    return data


def backtest_signals(signal_path='dataspace/dataset_signals.csv', 
                     signal_col='Signal', 
                     price_path='dataspace/dataset.csv', 
                     price_col='Close', 
                     initial_cash=100000, 
                     stats_output='dataspace/pf_stats.txt', 
                     plot_output='dataspace/pf_plot.html'):
    price_data = pd.read_csv(price_path, index_col=0, parse_dates=True)
    signals = pd.read_csv(signal_path, index_col=0, parse_dates=True)
    price_data = price_data.loc[signals.index]
    entries = signals[signal_col] == 1
    exits = signals[signal_col] == 0
    pf = vbt.Portfolio.from_signals(price_data[price_col], entries, exits, init_cash=initial_cash)
    stats = pf.stats()
    with open(stats_output, 'w') as f:
        f.write(stats.to_string())
    pf.plot().write_html(plot_output)
    return pf

import pandas as pd
import numpy as np

def mix_signal_strategy(signal1_path: str = 'dataspace/dataset_signals.csv',
                        signal2_path: str = 'dataspace/dataset_mfi_signals.csv',
                        signal1_name: str = 'Signal1',
                        signal2_name: str = 'Signal2',
                        output_file_path: str = 'dataspace/dataset_mixed_signals.csv',
                        buy_strategy: str = 'Signal1:1 AND Signal2:1',
                        sell_strategy: str = 'Signal1:-1 OR Signal2:-1'):
    """
    Generate trading signals by combining any two signal strategies.
    Users can specify any combination of conditions for buy and sell signals using a simple string format.

    :param signal1_path: Path to the CSV file containing the first signal. Default is 'dataspace/dataset_signals.csv'.
    :param signal2_path: Path to the CSV file containing the second signal. Default is 'dataspace/dataset_mfi_signals.csv'.
    :param signal1_name: Name to use for the first signal in strategy strings. Default is 'Signal1'.
    :param signal2_name: Name to use for the second signal in strategy strings. Default is 'Signal2'.
    :param output_file_path: Path to save the combined signals CSV file. Default is 'dataspace/dataset_mixed_signals.csv'.
    :param buy_strategy: Strategy for buy signals. Default is 'Signal1:1 AND Signal2:1'.
    :param sell_strategy: Strategy for sell signals. Default is 'Signal1:-1 OR Signal2:-1'.
    :return: pd.DataFrame with the combined trading signals.
    """
    # Load signals
    signal1_data = pd.read_csv(signal1_path, index_col=0, parse_dates=True)
    signal2_data = pd.read_csv(signal2_path, index_col=0, parse_dates=True)

    # Ensure both DataFrames have the same index
    common_index = signal1_data.index.intersection(signal2_data.index)
    signal1_data = signal1_data.loc[common_index]
    signal2_data = signal2_data.loc[common_index]

    # Create a new DataFrame for combined signals
    combined_signals = pd.DataFrame(index=common_index)
    combined_signals[signal1_name] = signal1_data['Signal']
    combined_signals[signal2_name] = signal2_data['Signal']

    def parse_strategy(strategy_str):
        conditions = strategy_str.split()
        parsed_condition = []
        for condition in conditions:
            if ':' in condition:
                signal, value = condition.split(':')
                parsed_condition.append(f"(combined_signals['{signal}'] == {value})")
            elif condition.upper() in ['AND', 'OR']:
                parsed_condition.append(condition.lower())
            else:
                parsed_condition.append(condition)
        return ' '.join(parsed_condition)

    buy_condition = parse_strategy(buy_strategy)
    sell_condition = parse_strategy(sell_strategy)

    # Generate combined signals based on the specified strategies
    combined_signals['Signal'] = np.select(
        [combined_signals.eval(buy_condition), combined_signals.eval(sell_condition)],
        [1, -1],
        default=0  # Do nothing if neither buy nor sell conditions are met
    )

    # Generate trading positions (changes in signal)
    combined_signals['Position'] = combined_signals['Signal'].diff()

    # Save combined signals to a CSV file
    combined_signals.to_csv(output_file_path)

    return combined_signals