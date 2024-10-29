import yfinance as yf
import pandas as pd
import json
from typing import Union, Dict, Any, List

def get_stock_info(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/stock_info.json') -> str:
    """
    Retrieves all stock information for a given ticker symbol using yfinance and saves it to a JSON file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the stock information will be saved. Defaults to 'dataspace/stock_info.json'.

    Returns:
    - str: The file path of the saved JSON file containing the stock information.
    """
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    with open(output_file_path, 'w') as f:
        json.dump(info, f)
    return output_file_path

def get_historical_market_data(ticker_symbol: str = 'MSFT', period: str = '1mo', output_file_path: str = 'dataspace/historical_data.csv') -> str:
    """
    Retrieves historical market data for a given ticker symbol over a specified period and saves it to a CSV file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - period (str): The period over which to retrieve data (e.g., '1d', '5d', '1mo', etc.). Defaults to '1mo'.
    - output_file_path (str): The file path where the historical data will be saved. Defaults to 'dataspace/historical_data.csv'.

    Returns:
    - str: The file path of the saved CSV file containing the historical market data.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    hist.to_csv(output_file_path)
    return output_file_path

def get_history_metadata(ticker_symbol: str = 'MSFT', period: str = '1mo', output_file_path: str = 'dataspace/history_metadata.json') -> str:
    """
    Retrieves metadata about the historical market data for a given ticker symbol and saves it to a JSON file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - period (str): The period over which to retrieve data to generate metadata. Defaults to '1mo'.
    - output_file_path (str): The file path where the metadata will be saved. Defaults to 'dataspace/history_metadata.json'.

    Returns:
    - str: The file path of the saved JSON file containing the history metadata.
    """
    ticker = yf.Ticker(ticker_symbol)
    ticker.history(period=period)
    metadata = ticker.history_metadata
    with open(output_file_path, 'w') as f:
        json.dump(metadata, f)
    return output_file_path

def get_actions(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/actions.xlsx') -> str:
    """
    Retrieves corporate actions like dividends, splits, and capital gains for a given ticker symbol and saves them to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the actions will be saved. Defaults to 'dataspace/actions.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the corporate actions.
    """
    ticker = yf.Ticker(ticker_symbol)
    actions = {
        'Actions': ticker.actions,
        'Dividends': ticker.dividends,
        'Splits': ticker.splits,
        'Capital_Gains': ticker.capital_gains
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in actions.items():
            df = df.to_frame() if isinstance(df, pd.Series) else df
            df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path

def get_share_count(ticker_symbol: str = 'MSFT', start_date: str = '2022-01-01', end_date: Union[str, None] = None, output_file_path: str = 'dataspace/share_count.csv') -> str:
    """
    Retrieves the share count over a specified time period for a given ticker symbol and saves it to a CSV file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - start_date (str): The start date in 'YYYY-MM-DD' format. Defaults to '2022-01-01'.
    - end_date (str): The end date in 'YYYY-MM-DD' format. Defaults to None (up to the current date).
    - output_file_path (str): The file path where the share count data will be saved. Defaults to 'dataspace/share_count.csv'.

    Returns:
    - str: The file path of the saved CSV file containing the share count data.
    """
    ticker = yf.Ticker(ticker_symbol)
    shares = ticker.get_shares_full(start=start_date, end=end_date)
    shares.to_csv(output_file_path)
    return output_file_path

def get_financials(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/financials.xlsx') -> str:
    """
    Retrieves various financial statements for a given ticker symbol and saves them to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the financial statements will be saved. Defaults to 'dataspace/financials.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the financial statements.
    """
    ticker = yf.Ticker(ticker_symbol)
    financials = {
        'Calendar': ticker.calendar,
        'SEC_Filings': ticker.sec_filings,
        'Income_Statement': ticker.income_stmt,
        'Quarterly_Income_Statement': ticker.quarterly_income_stmt,
        'Balance_Sheet': ticker.balance_sheet,
        'Quarterly_Balance_Sheet': ticker.quarterly_balance_sheet,
        'Cashflow': ticker.cashflow,
        'Quarterly_Cashflow': ticker.quarterly_cashflow
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in financials.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path

def get_holders(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/holders.xlsx') -> str:
    """
    Retrieves information about major holders, institutional holders, and insider transactions for a given ticker symbol and saves them to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the holders information will be saved. Defaults to 'dataspace/holders.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the holders information.
    """
    ticker = yf.Ticker(ticker_symbol)
    holders = {
        'Major_Holders': ticker.major_holders,
        'Institutional_Holders': ticker.institutional_holders,
        'Mutualfund_Holders': ticker.mutualfund_holders,
        'Insider_Transactions': ticker.insider_transactions,
        'Insider_Purchases': ticker.insider_purchases,
        'Insider_Roster_Holders': ticker.insider_roster_holders
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in holders.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path

def get_sustainability(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/sustainability.csv') -> str:
    """
    Retrieves sustainability data for a given ticker symbol and saves it to a CSV file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the sustainability data will be saved. Defaults to 'dataspace/sustainability.csv'.

    Returns:
    - str: The file path of the saved CSV file containing the sustainability data.
    """
    ticker = yf.Ticker(ticker_symbol)
    sustainability = ticker.sustainability
    if sustainability is not None and not sustainability.empty:
        sustainability.to_csv(output_file_path)
        return output_file_path
    else:
        return "No sustainability data available."

def get_recommendations(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/recommendations.xlsx') -> str:
    """
    Retrieves analyst recommendations and related data for a given ticker symbol and saves them to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the recommendations will be saved. Defaults to 'dataspace/recommendations.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the recommendations.
    """
    ticker = yf.Ticker(ticker_symbol)
    recommendations = {
        'Recommendations': ticker.recommendations,
        'Recommendations_Summary': ticker.recommendations_summary,
        'Upgrades_Downgrades': ticker.upgrades_downgrades
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in recommendations.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path

def get_analyst_data(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/analyst_data.xlsx') -> str:
    """
    Retrieves analyst estimates and historical data for a given ticker symbol and saves them to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the analyst data will be saved. Defaults to 'dataspace/analyst_data.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the analyst data.
    """
    ticker = yf.Ticker(ticker_symbol)
    analyst_data = {
        'Analyst_Price_Targets': ticker.analyst_price_targets,
        'Earnings_Estimate': ticker.earnings_estimate,
        'Revenue_Estimate': ticker.revenue_estimate,
        'Earnings_History': ticker.earnings_history,
        'EPS_Trend': ticker.eps_trend,
        'EPS_Revisions': ticker.eps_revisions,
        'Growth_Estimates': ticker.growth_estimates
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in analyst_data.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path

def get_earnings_dates(ticker_symbol: str = 'MSFT', limit: int = 12, output_file_path: str = 'dataspace/earnings_dates.csv') -> str:
    """
    Retrieves future and historical earnings dates for a given ticker symbol and saves them to a CSV file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - limit (int): The maximum number of earnings dates to retrieve. Defaults to 12.
    - output_file_path (str): The file path where the earnings dates will be saved. Defaults to 'dataspace/earnings_dates.csv'.

    Returns:
    - str: The file path of the saved CSV file containing the earnings dates.
    """
    ticker = yf.Ticker(ticker_symbol)
    earnings_dates = ticker.get_earnings_dates(limit=limit)
    earnings_dates.to_csv(output_file_path, index=False)
    return output_file_path

def get_isin(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/isin.txt') -> str:
    """
    Retrieves the ISIN (International Securities Identification Number) for a given ticker symbol and saves it to a text file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the ISIN will be saved. Defaults to 'dataspace/isin.txt'.

    Returns:
    - str: The file path of the saved text file containing the ISIN.
    """
    ticker = yf.Ticker(ticker_symbol)
    isin_code = ticker.isin
    with open(output_file_path, 'w') as f:
        f.write(isin_code)
    return output_file_path

def get_options_expirations(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/options_expirations.txt') -> str:
    """
    Retrieves available options expiration dates for a given ticker symbol and saves them to a text file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the options expirations will be saved. Defaults to 'dataspace/options_expirations.txt'.

    Returns:
    - str: The file path of the saved text file containing the options expiration dates.
    """
    ticker = yf.Ticker(ticker_symbol)
    expirations = ticker.options
    with open(output_file_path, 'w') as f:
        for date in expirations:
            f.write(f"{date}\n")
    return output_file_path

def get_news(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/news.json') -> str:
    """
    Retrieves recent news articles related to a given ticker symbol and saves them to a JSON file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - output_file_path (str): The file path where the news articles will be saved. Defaults to 'dataspace/news.json'.

    Returns:
    - str: The file path of the saved JSON file containing the news articles.
    """
    ticker = yf.Ticker(ticker_symbol)
    news = ticker.news
    with open(output_file_path, 'w') as f:
        json.dump(news, f)
    return output_file_path

def get_option_chain(ticker_symbol: str = 'MSFT', expiration_date: Union[str, None] = None, output_file_path: str = 'dataspace/option_chain.xlsx') -> str:
    """
    Retrieves the option chain data for a specific expiration date for a given ticker symbol and saves it to an Excel file in the dataspace folder.

    Parameters:
    - ticker_symbol (str): The stock ticker symbol to query. Defaults to 'MSFT'.
    - expiration_date (str): The expiration date in 'YYYY-MM-DD' format. Defaults to None (uses the earliest available date).
    - output_file_path (str): The file path where the option chain data will be saved. Defaults to 'dataspace/option_chain.xlsx'.

    Returns:
    - str: The file path of the saved Excel file containing the option chain data.
    """
    ticker = yf.Ticker(ticker_symbol)
    if expiration_date is None:
        expiration_date = ticker.options[0]
    opt = ticker.option_chain(expiration_date)
    option_data = {
        'Calls': opt.calls,
        'Puts': opt.puts
    }
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, df in option_data.items():
            df.to_excel(writer, sheet_name=sheet_name)
    return output_file_path


import os
import openai
import json
from typing import List, Dict, Optional, Union
import pandas as pd
import PyPDF2
from docx import Document
from openpyxl import load_workbook
from PIL import Image
import pytesseract
import csv

def universal_analyzer(
    input_path: str = 'dataspace/input/',
    output_file: str = 'dataspace/analysis_results.json',
    api_key: str = '',
    system_role: str = "You are a highly skilled analyst with expertise in multiple domains.",
    analysis_task: str = "Analyze this content and provide key insights, patterns, and recommendations.",
    additional_instructions: str = "",
    supported_file_types: List[str] = ['.txt', '.py', '.csv', '.json', '.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg'],
    model: str = "gpt-4",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    chunk_size: int = 4000  # Maximum characters per API call
) -> Dict:
    """
    A universal analyzer that can process various file types and provide AI-powered analysis using OpenAI.
    
    Parameters:
    -----------
    input_path : str
        Path to file or directory to analyze. Default is 'dataspace/input/'.
    output_file : str
        Path where analysis results will be saved as JSON. Default is 'dataspace/analysis_results.json'.
    api_key : str
        OpenAI API key for authentication. Required parameter.
    system_role : str
        Role description for the AI analyzer. Default is a general analyst role.
    analysis_task : str
        Specific task or question for analysis. Default is general analysis.
    additional_instructions : str
        Any additional instructions or context for the analysis.
    supported_file_types : List[str]
        List of supported file extensions. Default includes common file types.
    model : str
        OpenAI model to use. Default is "gpt-4".
    max_tokens : int
        Maximum tokens for OpenAI response. Default is 2000.
    temperature : float
        Temperature for OpenAI response (0.0 to 1.0). Default is 0.7.
    chunk_size : int
        Maximum characters per API call. Default is 4000.
        
    Returns:
    --------
    Dict
        Dictionary containing analysis results for each processed file.
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")

    openai.api_key = api_key
    analysis_results = {}

    def read_file_content(file_path: str) -> str:
        """Helper function to read various file types."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt' or file_extension == '.py':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
                    
            elif file_extension == '.pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return ' '.join([page.extract_text() for page in pdf_reader.pages])
                    
            elif file_extension == '.docx':
                doc = Document(file_path)
                return ' '.join([paragraph.text for paragraph in doc.paragraphs])
                
            elif file_extension == '.xlsx':
                workbook = load_workbook(filename=file_path)
                sheet = workbook.active
                return ' '.join([str(cell.value) for row in sheet.rows for cell in row if cell.value])
                
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                img = Image.open(file_path)
                return pytesseract.image_to_string(img)
                
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                return df.to_string()
                
            elif file_extension == '.json':
                with open(file_path, 'r') as file:
                    return json.dumps(json.load(file), indent=2)
                    
            else:
                return f"Unsupported file type: {file_extension}"
                
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def analyze_content(content: str, filename: str) -> str:
        """Helper function to analyze content using OpenAI API."""
        try:
            # Split content into chunks if it's too long
            content_chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            
            # Initialize conversation with system role
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"Task: {analysis_task}\n\nAdditional Instructions: {additional_instructions}\n\nAnalyzing file: {filename}\n"}
            ]
            
            full_analysis = []
            
            # Process each chunk
            for i, chunk in enumerate(content_chunks):
                chunk_messages = messages.copy()
                chunk_messages.append({
                    "role": "user",
                    "content": f"Content Part {i+1}/{len(content_chunks)}:\n{chunk}"
                })
                
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=chunk_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                full_analysis.append(response.choices[0].message.content)
            
            # If there were multiple chunks, summarize them
            if len(full_analysis) > 1:
                summary_messages = messages.copy()
                summary_messages.append({
                    "role": "user",
                    "content": f"Please provide a consolidated summary of all the previous analyses:\n\n{''.join(full_analysis)}"
                })
                
                summary_response = openai.ChatCompletion.create(
                    model=model,
                    messages=summary_messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                return summary_response.choices[0].message.content
            
            return full_analysis[0]
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Process single file or directory
        if os.path.isfile(input_path):
            file_extension = os.path.splitext(input_path)[1].lower()
            if file_extension in supported_file_types:
                content = read_file_content(input_path)
                analysis = analyze_content(content, os.path.basename(input_path))
                analysis_results[input_path] = {
                    'file_path': input_path,
                    'analysis': analysis
                }
        else:
            # Process directory
            for root, _, files in os.walk(input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()
                    
                    if file_extension in supported_file_types:
                        content = read_file_content(file_path)
                        analysis = analyze_content(content, file)
                        analysis_results[file] = {
                            'file_path': file_path,
                            'analysis': analysis
                        }

        # Save results to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)

        return analysis_results

    except Exception as e:
        error_result = {'error': str(e)}
        
        # Save error to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=4, ensure_ascii=False)
            
        return error_result
