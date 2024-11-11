from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Union
import joblib
# Import necessary libraries

def df_keep_rows_by_index(input_csv_file: str = 'dataspace/dataset.csv',
                       start_index: int = 0,
                       end_index: int = 100,
                       output_csv_file: str = 'dataspace/dataset_selected_rows.csv'):
    """
    Keeps rows in a pandas DataFrame based on specified index range.

    Parameters:
    input_csv_file (str): The file path to read the CSV file. Default is 'dataspace/dataset.csv'.
    start_index (int): The start index of the row range to keep.
    end_index (int): The end index of the row range to keep. This index is inclusive.
    output_csv_file (str): The file path to save the modified CSV file. Default is 'dataspace/selected_rows_dataset.csv'.

    Returns:
    pd.DataFrame: The DataFrame containing only the rows within the specified index range.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Validate index range
    if start_index < 0 or end_index >= len(df):
        raise ValueError("Start or end index is out of the DataFrame's range.")
    
    # Keep only the rows within the specified index range
    df = df.iloc[start_index:end_index + 1]  # +1 because the end index is inclusive
    
    # Save the modified DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)
    
    return df


def df_keep_columns(input_csv_file: str = 'dataspace/dataset.csv',
                          columns_to_keep: list = ['ImportantColumn1', 'ImportantColumn2'],
                          output_csv_file: str = 'dataspace/dataset_filtered.csv'):
    """
    Keeps specified columns in a pandas DataFrame and optionally filters rows based on a condition.

    Parameters:
    input_csv_file (str): The file path to read the CSV file. Default is 'dataspace/dataset.csv'.
    columns_to_keep (list): List of column names to be retained in the DataFrame.
                            Default is ['ImportantColumn1', 'ImportantColumn2'].
    output_csv_file (str): The file path to save the modified CSV file. Default is 'dataspace/filtered_dataset.csv'.

    Returns:
    pd.DataFrame: The DataFrame with only the specified columns and optionally filtered rows.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Check if all specified columns to keep are present in the DataFrame
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Keep only the specified columns
    df = df[columns_to_keep]
    
    # Save the modified DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)
    
    return df



def df_delete_columns(input_csv_file: str = 'dataspace/dataset.csv',
                   columns_to_delete: list = ['UnwantedColumn'],
                   output_csv_file: str = 'dataspace/dataset_cleaned.csv'):
    """
    Deletes specified columns from a pandas DataFrame.

    Parameters:
    input_csv_file (str): The file path to read the CSV file. Default is 'dataspace/dataset.csv'.
    columns_to_delete (list): List of column names to be deleted from the DataFrame.
                              Default is ['UnwantedColumn'].
    output_csv_file (str): The file path to save the modified CSV file. Default is 'dataspace/cleaned_dataset.csv'.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns removed.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Check if all specified columns to delete are present in the DataFrame
    missing_columns = [col for col in columns_to_delete if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Delete the specified columns
    df.drop(columns=columns_to_delete, inplace=True)
    
    # Save the modified DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)
    
    return df

def df_rename_columns(input_csv_file: str = 'dataspace/dataset.csv',
                      rename_dict: dict = {'Close': 'close', 'Open': 'open'},
                      output_csv_file: str = 'dataspace/dataset_renamed.csv'):
    """
    Renames multiple columns in a pandas DataFrame based on a dictionary mapping from old names to new names.

    Parameters:
    input_csv_file (str): The file path to read the CSV file. Default is 'dataspace/dataset.csv'.
    rename_dict (dict): Dictionary where keys are old column names and values are new column names.
                        Default is {'Close': 'close', 'Open': 'open'}.
    output_csv_file (str): The file path to save the modified CSV file. Default is 'dataspace/modified_dataset.csv'.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)
    
    # Check if all keys in the dictionary are present in the DataFrame columns
    missing_columns = [col for col in rename_dict if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")
    
    # Rename the columns
    df.rename(columns=rename_dict, inplace=True)
    
    # Save the modified DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)
    
    return df

def merge_csv_by_index(input_csv_file1: str = 'dataspace/dataset1.csv',
                       input_csv_file2: str = 'dataspace/dataset2.csv',
                       output_csv_file: str = 'dataspace/merged_dataset.csv',
                       rename_dict1: dict = None,
                       rename_dict2: dict = None,
                       how: str = 'outer') -> pd.DataFrame:
    """
    Merges two CSV files horizontally based on their index after optionally renaming columns in each.

    Parameters:
    input_csv_file1 (str): Path to the first CSV file. Default path is provided.
    input_csv_file2 (str): Path to the second CSV file. Default path is provided.
    output_csv_file (str): Path where the merged CSV file will be saved. Default path is provided.
    rename_dict1 (dict): Dictionary to rename columns of the first CSV file.
    rename_dict2 (dict): Dictionary to rename columns of the second CSV file.
    how (str): Type of merge to perform ('outer', 'inner', 'left', 'right'). Default is 'outer'.

    Returns:
    pd.DataFrame: A dataframe containing the merged data.
    """
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(input_csv_file1)
    df2 = pd.read_csv(input_csv_file2)

    # Rename columns if dictionaries are provided
    if rename_dict1:
        df1.rename(columns=rename_dict1, inplace=True)
    if rename_dict2:
        df2.rename(columns=rename_dict2, inplace=True)

    # Merge the DataFrames based on the index
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how=how)
    # Drop any 'Unnamed' columns that may exist
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(output_csv_file, index=False)

    return merged_df




def prepare_dataset_tabular_ml(input_file_path: str = 'dataspace/AirQuality.csv', 
                               index_col: Union[int, bool] = 0,
                               target_column_name: str = 'NO2(GT)', 
                               drop_columns: list = ['Date','NO2(GT)','Time'],
                               output_file_path:str = 'dataspace/prepared_dataset_tabular_ml.csv'):
    """
    Prepares a dataset for tabular machine learning by loading, cleaning, and optionally modifying it.
    """
    
    # Load the dataset
    df = pd.read_csv(input_file_path, index_col=index_col)
    
    # Drop any 'Unnamed' columns that may exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Set the target column if it exists
    if target_column_name in df.columns:
        df['target'] = df[target_column_name].values
    
    # Drop specified columns if any
    if drop_columns is not None:
        df = df.drop(columns=drop_columns, errors='ignore')
    
    # Save the cleaned dataset
    output_file_path = output_file_path
    df.to_csv(output_file_path)
    
    return df



def split_dataset_4ml(input_data_file:str = 'dataspace/prepared_dataset_tabular_ml.csv', 
                           index_col: Union[int, bool] = 0,
                           train_size:int =0.6, 
                           val_size:int =0.2, 
                           test_size:int =0.2,
                           output_train_file:str ='dataspace/train.csv', 
                           output_val_file:str ='dataspace/val.csv', 
                           output_test_file:str ='dataspace/test.csv'):
    """
    Split a dataset into training, validation, and test sets and save them as CSV files.
    
    Parameters:
    - data (DataFrame): The dataset to split.
    - train_size (float): The proportion of the dataset to include in the train split.
    - val_size (float): The proportion of the dataset to include in the validation split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - train_path (str): File path to save the training set CSV.
    - val_path (str): File path to save the validation set CSV.
    - test_path (str): File path to save the test set CSV.
    
    Returns:
    - train_data (DataFrame): The training set.
    - val_data (DataFrame): The validation set.
    - test_data (DataFrame): The test set.
    """

    # Load data
    data = pd.read_csv(input_data_file,index_col=index_col)

    if not (0 < train_size < 1) or not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("All size parameters must be between 0 and 1.")
        
    if train_size + val_size + test_size != 1:
        raise ValueError("The sum of train_size, val_size, and test_size must equal 1.")
        
    # First split to separate out the test set
    temp_train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Adjust val_size proportion to account for the initial split (for the remaining data)
    adjusted_val_size = val_size / (1 - test_size)
    
    # Second split to separate out the validation set from the temporary training set
    train_data, val_data = train_test_split(temp_train_data, test_size=adjusted_val_size, random_state=42)
    
    # Save to CSV
    train_data.to_csv(output_train_file, index=False)
    val_data.to_csv(output_val_file, index=False)
    test_data.to_csv(output_test_file, index=False)
    
    return train_data, val_data, test_data


def df_shift_data_row(input_csv_file: str = 'dataspace/dataset.csv',
                      shift_columns: list = ['Close'],
                      shift_columns_name: list = ['Close_shifted'],
                      shift_columns_keep: bool = False,
                      shift_rows: int = -1,
                      keep_row: bool = False,
                      output_csv_file: str = 'dataspace/dataset_selected_rows.csv'):
    """
    Shifts the data in specified columns by the specified number of rows and optionally renames and retains original columns.

    Args:
    input_csv_file (str): Path to the input CSV file.
    shift_columns (list): List of column names whose data will be shifted.
    shift_columns_name (list): List of new names for the shifted columns.
    shift_columns_keep (bool): If True, keeps the original columns alongside the shifted ones.
    shift_rows (int): Number of rows to shift the data by. 1 is one day delay, -1 is one day ahead
    keep_row (bool): If False, rows resulting in NaN values from the shift will be dropped.
    output_csv_file (str): Path to save the output CSV file after shifting the data.
    
    Returns:
    pd.DataFrame: DataFrame with the data shifted and potentially renamed in specified columns.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file)

    # Shift the data in the specified columns and optionally rename
    for orig, new in zip(shift_columns, shift_columns_name):
        df[new] = df[orig].shift(periods=shift_rows)
        if not shift_columns_keep:
            df.drop(orig, axis=1, inplace=True)

    # If keep_row is False, drop rows with NaN values that result from shifting
    if not keep_row:
        df = df.dropna(subset=shift_columns_name)

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv_file, index=False)

    return df









