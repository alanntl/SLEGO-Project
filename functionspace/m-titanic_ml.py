import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Microservice: Preprocess data

def m_preprocess_titanic_data(
    train_data_path: str = './dataspace/train.csv',
    test_data_path: str = './dataspace/test.csv',
    target_column: str = 'Survived',
    drop_columns: list = ['Name', 'Ticket', 'Cabin'],
    categorical_features: list = ['Sex', 'Embarked'],
    numerical_features: list = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'],
    numerical_impute_strategy: str = 'mean',
    categorical_impute_strategy: str = 'most_frequent',
    test_size: float = 0.2,
    random_state: int = 42):
    """
    Preprocess Titanic data by handling missing values and encoding features.

    Parameters:
    train_data_path (str): Path to the training dataset CSV file.
    test_data_path (str): Path to the testing dataset CSV file.
    target_column (str): Name of the target column.
    drop_columns (list): List of columns to drop from the dataset.
    categorical_features (list): List of categorical feature column names.
    numerical_features (list): List of numerical feature column names.
    numerical_impute_strategy (str): Strategy for imputing numerical features.
    categorical_impute_strategy (str): Strategy for imputing categorical features.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series: Preprocessed training features, validation features, test features, and training labels.
    """
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    X = train_data.drop([target_column] + drop_columns, axis=1)
    y = train_data[target_column]
    X_test = test_data.drop(drop_columns, axis=1)

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=numerical_impute_strategy)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X = preprocessor.fit_transform(X)
    X_test = preprocessor.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val
