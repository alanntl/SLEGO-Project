import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def common_statistics(input_file: str = "dataspace/apple_dataset.csv", output_file: str = "dataspace/minhua_trial.csv", columns: list[str] = []):
    """
    Computes common statistics for the dataset and saves them to a CSV file.

    Parameters:
    input_file (str): Path to the input CSV dataset.
    output_file (str): Path where the output CSV file will be saved.
    columns (list): List of columns to compute statistics for. If empty, all columns are used.

    Returns:
    dict: A dictionary containing the computed statistics.
    """

# Load the data from the CSV file
    data = pd.read_csv(input_file)

    if columns:
        data = data[columns]
    else:
        # Ensure 'Date' is retained
        data = data.drop(columns=['Dividends', 'Stock Splits', 'Datetime'])

    # Exclude 'Date' from statistical calculations
    numerical_data = data.select_dtypes(include=[np.number])

    # Compute the statistics
    mean = numerical_data.mean()
    median = numerical_data.median()
    variance = numerical_data.var()
    std_dev = numerical_data.std()

    # Prepare the statistics DataFrame
    stats = pd.DataFrame(columns=data.columns)
    stats = stats.append(pd.Series(name='Mean'), ignore_index=True)
    stats = stats.append(pd.Series(name='Median'), ignore_index=True)
    stats = stats.append(pd.Series(name='Variance'), ignore_index=True)
    stats = stats.append(pd.Series(name='Std Dev'), ignore_index=True)

    # Assign the computed statistics to the respective columns, excluding 'Date'
    for col in numerical_data.columns:
        stats.at[0, col] = mean[col]
        stats.at[1, col] = median[col]
        stats.at[2, col] = variance[col]
        stats.at[3, col] = std_dev[col]

    # Append the original data with statistics
    final_data = pd.concat([data, stats], ignore_index=True)

    # Save the statistics to a CSV file
    final_data.to_csv(output_file, index=False)

    # Return the computed statistics in a dataframe
    return final_data


def moving_average(input_file: str = "dataspace/minhua_trial.csv", output_file: str = "dataspace/minhua_moving_average.csv", window_size: int = 3):
    """
    Computes the moving average for the dataset and saves it to a CSV file.

    Parameters:
    input_file (str): Path to the input CSV dataset.
    output_file (str): Path where the output CSV file will be saved.
    window_size (int): Size of the moving average window.

    Returns:
    pd.DataFrame: A dataframe containing the moving average values.
    """

    # Load the data from the CSV file
    data = pd.read_csv(input_file)

    data = data.drop(columns=['Date', 'Dividends', 'Stock Splits', 'Datetime'])

    # Compute the moving average
    moving_avg = data.rolling(window=window_size).mean()

    # Save the moving average to a CSV file
    moving_avg.to_csv(output_file, index=False)

    # Return the moving average values
    return moving_avg

def plot_prices(input_file: str = "dataspace/minhua_trial.csv", output_file: str = "dataspace/minhua_plot.png"):
    """
    Plots the stock prices from the dataset and saves the plot as an image.

    Parameters:
    input_file (str): Path to the input CSV dataset.
    output_file (str): Path where the output plot image will be saved.

    Returns:
    str: Path to the saved plot image.
    """

    # Load the data from the CSV file
    data = pd.read_csv(input_file)

    data['Date'] = pd.to_datetime(data['Date'])

    data = data.drop(columns=['Dividends', 'Stock Splits', 'Datetime'])

    fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    
    # Plot Open
    axs[0].plot(data['Date'], data['Open'], label='Open', color='blue')
    axs[0].set_title('Open Prices')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    
    # Plot High
    axs[1].plot(data['Date'], data['High'], label='High', color='green')
    axs[1].set_title('High Prices')
    axs[1].set_ylabel('Price')
    axs[1].legend()
    
    # Plot Low
    axs[2].plot(data['Date'], data['Low'], label='Low', color='red')
    axs[2].set_title('Low Prices')
    axs[2].set_ylabel('Price')
    axs[2].legend()
    
    # Plot Close
    axs[3].plot(data['Date'], data['Close'], label='Close', color='purple')
    axs[3].set_title('Close Prices')
    axs[3].set_ylabel('Price')
    axs[3].legend()
    
    # Plot Volume
    axs[4].plot(data['Date'], data['Volume'], label='Volume', color='orange')
    axs[4].set_title('Volume')
    axs[4].set_xlabel('Date')
    axs[4].set_ylabel('Volume')
    axs[4].legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file)

    # Return the path to the saved plot image
    # return output_file