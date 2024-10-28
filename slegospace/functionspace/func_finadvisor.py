import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from typing import List, Union
import json
import datetime

def collect_fundamental_data(
    tickers: List[str] = ['AAPL', 'MSFT'], 
    output_file_path: str = 'dataspace/fundamental_data.json'
) -> dict:
    """
    Collects fundamental financial data for given stock tickers using Yahoo Finance.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols. Default is ['AAPL', 'MSFT'].
    - output_file_path (str): Path to save the resulting JSON file containing fundamental data.

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
    - start_date (str): The start date for historical data in 'YYYY-MM-DD' format.
    - end_date (str): The end date for historical data in 'YYYY-MM-DD' format.
    - output_file_path (str): Path to save the resulting CSV file.

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

def collect_finance_news(
    tickers: List[str] = ['AAPL', 'MSFT'], 
    max_articles: int = 5, 
    output_file_path: str = 'dataspace/finance_news.json'
) -> List[dict]:
    """
    Collects recent financial news using web scraping for given tickers.

    Parameters:
    - tickers (List[str]): A list of stock ticker symbols.
    - max_articles (int): Maximum number of news articles to fetch per ticker.
    - output_file_path (str): Path to save the resulting JSON file.

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
    - tickers (List[str]): A list of stock ticker symbols.
    - fundamentals (dict): Dictionary containing fundamental data.
    - historical_data (pd.DataFrame): DataFrame containing historical stock data.
    - news (list): List of news articles.
    - openai_api_key (str): OpenAI API key for authentication.
    - model_name (str): Name of the OpenAI model to use.
    - output_file_path (str): Path to save the generated report.

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

    # Recent price trends from historical data
    if not historical_data.empty:
        report_summary += "\nRecent Price Trends:\n"
        for ticker in tickers:
            ticker_data = historical_data[historical_data['Ticker'] == ticker].tail(5)
            report_summary += f"\n{ticker} last 5 days:\n{ticker_data[['Datetime', 'Close']].to_string()}\n"

    # Prepare the prompt for GPT
    prompt = f"""
You are a financial advisor. Based on the following data, provide a comprehensive analysis and investment recommendation:

{report_summary}

Please structure your analysis as follows:
1. Market Overview
2. Company Analysis (for each company)
3. Risk Factors
4. Investment Recommendations
5. Potential Catalysts to Watch

Focus on key metrics, recent developments, and potential future catalysts. Provide specific, actionable recommendations.
"""

    # Generate report using OpenAI API
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7
    )

    report_text = response['choices'][0]['message']['content']

    # Save the report to a file
    with open(output_file_path, 'w') as f:
        f.write(report_text)

    return report_text

def run_financial_analysis(
    tickers: List[str] = ['AAPL', 'MSFT'],
    start_date: str = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d"),
    end_date: str = datetime.datetime.now().strftime("%Y-%m-%d"),
    max_articles: int = 5,
    openai_api_key: str = '',
    model_name: str = 'gpt-3.5-turbo'
) -> str:
    """
    Runs the complete financial analysis pipeline: collects all data and generates an analysis report.

    Parameters:
    - tickers (List[str]): List of stock tickers to analyze
    - start_date (str): Start date for historical data
    - end_date (str): End date for historical data
    - max_articles (int): Maximum number of news articles per ticker
    - openai_api_key (str): OpenAI API key
    - model_name (str): OpenAI model to use

    Returns:
    - str: The generated analysis report
    """
    # Step 1: Collect fundamental data
    print("Collecting fundamental data...")
    fundamentals = collect_fundamental_data(tickers)
    
    # Step 2: Collect historical data
    print("Collecting historical data...")
    historical_data = collect_historical_data(tickers, start_date, end_date)
    
    # Step 3: Collect news
    print("Collecting recent news...")
    news = collect_finance_news(tickers, max_articles)
    
    # Step 4: Generate report
    print("Generating analysis report...")
    report = generate_advisor_report(
        tickers=tickers,
        fundamentals=fundamentals,
        historical_data=historical_data,
        news=news,
        openai_api_key=openai_api_key,
        model_name=model_name
    )
    
    print("Analysis complete! Report has been generated.")
    return report