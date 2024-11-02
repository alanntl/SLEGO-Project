# clustering_pipeline.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import plotly.express as px

def fetch_iris_dataset(output_csv_file: str = 'dataspace/iris_dataset.csv'):
    """
    Fetches the Iris dataset and saves it as a CSV file.

    Parameters:
    - output_csv_file (str): Path to save the dataset as a CSV file.

    Returns:
    - pd.DataFrame: The dataset as a pandas DataFrame.
    """
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename('target')], axis=1)
    df.to_csv(output_csv_file, index=False)
    return df

def perform_kmeans_clustering(
    input_file_path: str = 'dataspace/iris_dataset.csv',
    n_clusters: int = 3,
    output_file_path: str = 'dataspace/iris_clusters.csv'
):
    """
    Performs KMeans clustering on the dataset.

    Parameters:
    - input_file_path (str): Path to the dataset CSV file.
    - n_clusters (int): Number of clusters.
    - output_file_path (str): Path to save the clustered data.

    Returns:
    - pd.DataFrame: DataFrame with cluster labels.
    """
    df = pd.read_csv(input_file_path)
    X = df.drop(columns=['target'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    df.to_csv(output_file_path, index=False)
    return df

def plotly_cluster_visualization(
    input_csv_file: str = 'dataspace/iris_clusters.csv',
    x_axis: str = 'sepal length (cm)',
    y_axis: str = 'sepal width (cm)',
    cluster_col: str = 'Cluster',
    output_html_file: str = 'dataspace/iris_clusters_plot.html'
):
    """
    Visualizes clusters using Plotly.

    Parameters:
    - input_csv_file (str): Path to the CSV file with clustered data.
    - x_axis (str): Column name for the X-axis.
    - y_axis (str): Column name for the Y-axis.
    - cluster_col (str): Column name for cluster labels.
    - output_html_file (str): Path to save the HTML plot.

    Returns:
    - None
    """
    df = pd.read_csv(input_csv_file)
    fig = px.scatter(df, x=x_axis, y=y_axis, color=cluster_col)
    fig.write_html(output_html_file)