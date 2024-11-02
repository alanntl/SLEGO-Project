import pandas as pd
from autogluon.tabular import TabularPredictor
import pickle

def train_autogluon_tabular(input_file_path:str='dataspace/train.csv',
                               target:str='target',
                               train_frac:float=0.8,
                               random_state:int=42,
                               performance_output_path:str='dataspace/performance_autogluon.txt',
                               model_save_path:str='dataspace/autogluon_model.pkl'):
    """
    Trains a classification model using AutoGluon on the specified dataset and saves the trained model.

    Parameters:
        input_file_path (str): Path to the CSV file containing the dataset.
        target (str): Name of the column to predict.
        train_frac (float): Fraction of the dataset to use for training.
        random_state (int): Seed for the random number generator for reproducibility.
        performance_output_path (str): Path to save the text file containing model performance metrics.
        model_save_path (str): Path to save the trained AutoGluon model.

    Returns:
        Tuple[TabularPredictor, dict]: A tuple containing the trained AutoGluon TabularPredictor and 
                                       a dictionary with performance metrics.

    Saves the trained model using AutoGluon's built-in save mechanism and optionally as a pickle file.
    Evaluates the model's performance and writes metrics to the specified text file.
    """  
    # Load the dataset from a CSV file
    df = pd.read_csv(input_file_path)
    df = df.loc[:, ~df.columns.str.contains('Unnamed: 0', case=False)]
    train_data = df.sample(frac=train_frac, random_state=random_state)
    test_data = df.drop(train_data.index)

    # Train a classifier with AutoGluon
    predictor = TabularPredictor(label=target).fit(train_data)
    performance = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data, silent=True)

    # Write the performance metrics and leaderboard to a file
    with open(performance_output_path, 'w') as f:
        f.write(str(performance))
        f.write("\n")
        f.write(str(leaderboard))

    # Save the trained model using AutoGluon's method
    predictor.save(model_save_path)

    # Optionally, save the model using pickle
    pickle_path = model_save_path
    with open(pickle_path, 'wb') as f:
        pickle.dump(predictor, f)

    return predictor, performance

def autogluon_model_predict(pickle_path:str= 'dataspace/autogluon_model.pkl',
                        input_data_path:str='dataspace/test.csv',
                        output_data_path:str='dataspace/autogluon_predict.csv' ):
    """
    Loads a pickled AutoGluon model from the specified path.

    Parameters:
        pickle_path (str): Path to the pickled model file.

    Returns:
        TabularPredictor: The loaded AutoGluon model.

    Note:
        Loading models via pickle can lead to issues if there are mismatches in library versions or 
        if the saved model includes elements that pickle cannot handle properly. It is generally 
        recommended to use AutoGluon's native load functionality unless there is a specific need 
        for pickle.
    """

    with open(pickle_path, 'rb') as f:
        loaded_model = pickle.load(f)

    predictions = loaded_model.predict(input_data_path)
        # Save the predictions to a CSV file
    predictions.to_csv(output_data_path, index=False)


    return predictions

