import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from typing import List, Union
import json
import datetime

'''
Guidelines for building microservices (python functions):

1. Create a python function
2. Make sure the function has a docstring that explain to user how to use the microservice, and what the service does
3. Make sure the function has a return statement
4. Make sure the function has a parameter
5. Make sure the function has a default value for the parameter
6. Make sure the function has a type hint for the parameter 
'''

# Step 1: Collect Fundamental Data from Yahoo Finance
def collect_fundamental_data(
    tickers: List[str] = ['AAPL', 'MSFT'], 
    output_file_path: str = 'dataspace/fundamental_data.json'
) -> dict:
    """
    Collects fundamental financial data for given stock tickers using Yahoo Finance.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - output_file_path (str): Path to save the resulting JSON file containing fundamental data. Default is 'dataspace/fundamental_data.json'.

    Returns:
    - dict: A dictionary containing fundamental data for the tickers.
    """
    all_fundamentals = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        fundamentals = {
            'info': stock.info,
            'balance_sheet': stock.balance_sheet.to_dict() if not stock.balance_sheet.empty else {},
            'cashflow': stock.cashflow.to_dict() if not stock.cashflow.empty else {},
            'earnings': stock.earnings.to_dict() if not stock.earnings.empty else {},
            'financials': stock.financials.to_dict() if not stock.financials.empty else {},
        }
        all_fundamentals[ticker] = fundamentals

    with open(output_file_path, 'w') as f:
        json.dump(all_fundamentals, f)
    return all_fundamentals

# Step 2: Collect Historical Stock Data
def collect_historical_data(
    tickers: List[str] = ['AAPL', 'MSFT'], 
    start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"), 
    end_date: str = datetime.datetime.now().strftime("%Y-%m-%d"), 
    output_file_path: str = 'dataspace/historical_data.csv'
) -> pd.DataFrame:
    """
    Collects historical stock data for given stock tickers.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - start_date (str): The start date for historical data in 'YYYY-MM-DD' format. Default is one year ago from today.
    - end_date (str): The end date for historical data in 'YYYY-MM-DD' format. Default is today.
    - output_file_path (str): Path to save the resulting CSV file containing historical data. Default is 'dataspace/historical_data.csv'.

    Returns:
    - pd.DataFrame: DataFrame containing historical stock data for all tickers.
    """
    historical_data = pd.DataFrame()
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        data['Ticker'] = ticker
        data['Datetime'] = data.index
        historical_data = pd.concat([historical_data, data], ignore_index=True)
    
    historical_data.to_csv(output_file_path, index=False)
    return historical_data

# Step 3: Collect Finance News
def collect_finance_news(
    tickers: List[str] = ['AAPL', 'MSFT'], 
    max_articles: int = 5, 
    output_file_path: str = 'dataspace/finance_news.json'
) -> List[dict]:
    """
    Collects recent financial news using web scraping for given tickers.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - max_articles (int): Maximum number of news articles to fetch per ticker. Default is 5.
    - output_file_path (str): Path to save the resulting JSON file containing news. Default is 'dataspace/finance_news.json'.

    Returns:
    - list: A list of dictionaries containing news headlines and summaries.
    """
    all_news = []
    for ticker in tickers:
        url = f"https://news.google.com/search?q={ticker}%20finance&hl=en-US&gl=US&ceid=US%3Aen"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article', limit=max_articles)

        news_list = []
        for article in articles:
            headline = article.find('h3').text if article.find('h3') else 'No Headline'
            summary_tag = article.find('span')
            summary = summary_tag.text if summary_tag else 'No Summary'
            news_list.append({'ticker': ticker, 'headline': headline, 'summary': summary})
        all_news.extend(news_list)
    
    with open(output_file_path, 'w') as f:
        json.dump(all_news, f)
    return all_news

# Step 4: Generate Financial Advisor Report using GPT
def generate_advisor_report(
    tickers: List[str] = ['AAPL', 'MSFT'],
    fundamentals: dict = None,
    historical_data: pd.DataFrame = None,
    news: List[dict] = None,
    openai_api_key: str = '',
    model_name: str = 'gpt-3.5-turbo',
    output_file_path: str = 'dataspace/advisor_report.txt'
) -> str:
    """
    Generates a financial advisor report using OpenAI's GPT by analyzing collected data.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - fundamentals (dict): Dictionary containing fundamental data. If None, it will be collected.
    - historical_data (pd.DataFrame): DataFrame containing historical stock data. If None, it will be collected.
    - news (list): List of news articles. If None, it will be collected.
    - openai_api_key (str): OpenAI API key for authentication.
    - model_name (str): Name of the OpenAI model to use. Default is 'gpt-3.5-turbo'.
    - output_file_path (str): Path to save the generated report. Default is 'dataspace/advisor_report.txt'.

    Returns:
    - str: The generated financial advisor report.
    """
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Collect data if not provided
    if fundamentals is None:
        fundamentals = collect_fundamental_data(tickers)
    if historical_data is None:
        historical_data = collect_historical_data(tickers)
    if news is None:
        news = collect_finance_news(tickers)

    # Combine all information into a summary
    report_summary = ''
    for ticker in tickers:
        data = fundamentals.get(ticker, {})
        report_summary += f"""
Ticker: {ticker}
Fundamental Data: {data.get('info', {}).get('longBusinessSummary', 'No Summary Available')}
Balance Sheet (Snippet): {pd.DataFrame.from_dict(data.get('balance_sheet', {})).iloc[:, :2].to_string() if data.get('balance_sheet', {}) else 'No Balance Sheet Data'}
Recent News:
"""
        ticker_news = [article for article in news if article['ticker'] == ticker]
        for idx, article in enumerate(ticker_news):
            report_summary += f"{idx + 1}. {article['headline']}: {article['summary']}\n"

    # Prepare the prompt for GPT
    prompt = f"""
You are a financial advisor. Based on the data below, provide an analysis and financial recommendation for an investor:
{report_summary}
"""

    # Generate report using OpenAI API
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    report_text = response['choices'][0]['message']['content']

    # Save the report to a file
    with open(output_file_path, 'w') as f:
        f.write(report_text)

    return report_text

# Example of using the full pipeline
def financial_advisor_pipeline(
    tickers: List[str] = ['AAPL', 'MSFT'],
    start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
    end_date: str = datetime.datetime.now().strftime("%Y-%m-%d"),
    max_articles: int = 5,
    openai_api_key: str = '',
    model_name: str = 'gpt-3.5-turbo',
    fundamental_data_output_file: str = 'dataspace/fundamental_data.json',
    historical_data_output_file: str = 'dataspace/historical_data.csv',
    finance_news_output_file: str = 'dataspace/finance_news.json',
    advisor_report_output_file: str = 'dataspace/advisor_report.txt'
) -> str:
    """
    Executes the full financial advisor pipeline: collects fundamental data, historical data, news, and generates a report.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - start_date (str): The start date for historical data in 'YYYY-MM-DD' format. Default is one year ago from today.
    - end_date (str): The end date for historical data in 'YYYY-MM-DD' format. Default is today.
    - max_articles (int): Maximum number of news articles to fetch per ticker. Default is 5.
    - openai_api_key (str): OpenAI API key for authentication.
    - model_name (str): Name of the OpenAI model to use. Default is 'gpt-3.5-turbo'.
    - fundamental_data_output_file (str): Path to save the fundamental data JSON file. Default is 'dataspace/fundamental_data.json'.
    - historical_data_output_file (str): Path to save the historical data CSV file. Default is 'dataspace/historical_data.csv'.
    - finance_news_output_file (str): Path to save the finance news JSON file. Default is 'dataspace/finance_news.json'.
    - advisor_report_output_file (str): Path to save the generated advisor report. Default is 'dataspace/advisor_report.txt'.

    Returns:
    - str: The generated financial advisor report.
    """
    # Collect Fundamental Data
    fundamentals = collect_fundamental_data(
        tickers=tickers, 
        output_file_path=fundamental_data_output_file
    )

    # Collect Historical Stock Data
    historical_data = collect_historical_data(
        tickers=tickers, 
        start_date=start_date, 
        end_date=end_date, 
        output_file_path=historical_data_output_file
    )

    # Collect Finance News
    news = collect_finance_news(
        tickers=tickers, 
        max_articles=max_articles, 
        output_file_path=finance_news_output_file
    )

    # Generate Financial Advisor Report
    advisor_report = generate_advisor_report(
        tickers=tickers,
        fundamentals=fundamentals,
        historical_data=historical_data,
        news=news,
        openai_api_key=openai_api_key,
        model_name=model_name,
        output_file_path=advisor_report_output_file
    )

    return advisor_report
