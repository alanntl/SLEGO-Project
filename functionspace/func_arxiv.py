import arxiv
import csv
from typing import List, Dict
import requests
import pandas as pd

import pandas as pd
import requests
import os

def download_papers_from_arxiv_csv(filename: str = "dataspace/latest_papers.csv", 
                                   download_folder: str = "dataspace/papers/",
                                   url_col: str = "entry_id",
                                   title_col: str = "title"):
    """
    Download papers from arXiv based on a CSV file.

    Parameters:
        filename (str): Path to the CSV file with arXiv paper details.
        download_folder (str): Folder to save downloaded papers.
        url_col (str): Column name in CSV that contains the arXiv URL.
        title_col (str): Column name in CSV that contains the paper title.
    """
    # Ensure the directory exists
    os.makedirs(download_folder, exist_ok=True)
    
    # Read the CSV file using Pandas
    df = pd.read_csv(filename)
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        arxiv_url = row[url_col]
        title = row[title_col].replace('/', '_').replace(':', '_')  # Clean title for filename
        arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0]  # Extract arXiv ID and remove version

        # Format the download URL and filename
        download_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        file_path = os.path.join(download_folder, f"{title}.pdf")

        # Download the paper
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download {title} with ID {arxiv_id}")

    return 'Download finished, please check your folder!'


def search_arxiv_papers(search_query: str = 'machine learning', 
                        filename: str = "dataspace/latest_papers.csv", 
                        max_results: int = 5,
                        sort_by: str = "submitted",
                        sort_order: str = "descending"):
    """
    Searches for papers on arXiv, saves the results to a CSV file, and allows sorting of the results.

    Args:
    search_query (str): The query term to search for on arXiv.
    filename (str): Path to save the CSV file containing the search results.
    max_results (int): Maximum number of results to fetch and save.
    sort_by (str): Criterion to sort the search results by ("relevance", "lastUpdatedDate", "submitted").
    sort_order (str): Order to sort the search results ("ascending", "descending").

    Returns:
    DataFrame: A pandas DataFrame containing details of the fetched papers.
    """

    # Map user-friendly sorting terms to arXiv API's SortCriterion and SortOrder
    sort_options = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submitted": arxiv.SortCriterion.SubmittedDate
    }
    order_options = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending
    }

    # Fetch the results using the arXiv API
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_options.get(sort_by, arxiv.SortCriterion.SubmittedDate),
        sort_order=order_options.get(sort_order, arxiv.SortOrder.Descending)
    )
    results = list(search.results())

    # Convert results to DataFrame
    data = []
    for result in results:
        entry = {
            "entry_id": result.entry_id,
            "updated": result.updated.isoformat(),
            "published": result.published.isoformat(),
            "title": result.title,
            "authors": ', '.join([author.name for author in result.authors]),
            "summary": result.summary.replace('\n', ' '),
            "comment": result.comment,
            "journal_ref": result.journal_ref,
            "doi": result.doi,
            "primary_category": result.primary_category,
            "categories": ', '.join(result.categories)
        }
        data.append(entry)
    df = pd.DataFrame(data)

    # Save the DataFrame to CSV
    df.to_csv(filename, index=False)

    return df