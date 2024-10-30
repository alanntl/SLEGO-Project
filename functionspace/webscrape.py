import requests
from bs4 import BeautifulSoup

def webscrape_to_txt(url: str = "https://au.finance.yahoo.com/", 
                            output_filename: str = "dataspace/output_webscrape.txt"):
    """
    Fetches the content from the specified URL and saves the textual content 
    into a text file, stripping out all HTML tags.

    Parameters:
    - url (str): The URL from which to fetch the content. Default is Yahoo Finance homepage.
    - output_filename (str): The path to the file where the text content will be saved.

    Returns:
    - None: Outputs a file with the extracted text content.
    """
    try:
        # Send a HTTP request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text using .get_text()
            text_content = soup.get_text(separator='\n', strip=True)
            # Open a file in write mode
            with open(output_filename, 'w', encoding='utf-8') as file:
                file.write(text_content)
            print(f"Text content saved successfully to {output_filename}")
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

        return text_content
    except Exception as e:
        print(f"An error occurred: {e}")

