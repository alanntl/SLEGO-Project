# sentiment_analysis_pipeline.py

import pandas as pd
import nltk
import joblib
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def fetch_nltk_movie_reviews(output_csv_file: str = 'dataspace/movie_reviews.csv'):
    """
    Fetches the NLTK movie reviews dataset and saves it as a CSV file.

    Parameters:
    - output_csv_file (str): Path to save the dataset.

    Returns:
    - pd.DataFrame: DataFrame containing reviews and labels.
    """
    nltk.download('movie_reviews', quiet=True)

    documents = [
        (' '.join(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]

    reviews = [doc for doc, _ in documents]
    labels = [label for _, label in documents]

    df = pd.DataFrame({'Review': reviews, 'Label': labels})
    df.to_csv(output_csv_file, index=False)
    return df

def train_text_classification_model(
    input_file_path: str = 'dataspace/movie_reviews.csv',
    text_column: str = 'Review',
    target_column: str = 'Label',
    test_size: float = 0.2,
    random_state: int = 42,
    model_output_path: str = 'dataspace/text_classification_model.pkl',
    performance_output_path: str = 'dataspace/text_classification_performance.txt'
):
    """
    Trains a text classification model using Logistic Regression.

    Parameters:
    - input_file_path (str): Path to the dataset CSV file.
    - text_column (str): Name of the text column.
    - target_column (str): Name of the target column.
    - test_size (float): Fraction of data to use as test set.
    - random_state (int): Seed for reproducibility.
    - model_output_path (str): Path to save the trained model.
    - performance_output_path (str): Path to save performance metrics.

    Returns:
    - dict: Dictionary containing performance metrics.
    """
    df = pd.read_csv(input_file_path)
    X = df[text_column]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vect, y_train)

    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save the model and vectorizer
    joblib.dump((model, vectorizer), model_output_path)

    # Save performance metrics
    with open(performance_output_path, 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Classification Report:\n{report}')

    return {'Accuracy': accuracy, 'Report': report}

def text_model_predict(
    model_path: str = 'dataspace/text_classification_model.pkl',
    input_texts: list = [
        "I absolutely loved this movie. The story was gripping and the characters were well-developed.",
        "This was the worst film I have ever seen. Completely a waste of time."
    ],
    output_file_path: str = 'dataspace/text_predictions.csv'
):
    """
    Uses the trained text classification model to predict sentiments of new texts.

    Parameters:
    - model_path (str): Path to the saved model and vectorizer.
    - input_texts (list): List of texts to classify.
    - output_file_path (str): Path to save the predictions.

    Returns:
    - pd.DataFrame: DataFrame containing texts and their predicted labels.
    """
    model, vectorizer = joblib.load(model_path)
    X_new_vect = vectorizer.transform(input_texts)
    predictions = model.predict(X_new_vect)

    df = pd.DataFrame({'Text': input_texts, 'Predicted_Label': predictions})
    df.to_csv(output_file_path, index=False)
    return df