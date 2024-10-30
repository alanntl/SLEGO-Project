# Import required libraries
import os
import json
import pandas as pd
from typing import Union, Dict, Any, List
from openai import OpenAI
from bs4 import BeautifulSoup
import yfinance as yf

# =====================================
# Internal Utility Functions
# =====================================

def __check_module(module_name: str) -> bool:
    """Private function to check if a Python module is installed."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None

def __ensure_directory_exists(file_path: str) -> None:
    """Private function to ensure directory exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def __extract_text_from_html(html_content: str) -> str:
    """Private function to extract text from HTML content."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        raise ValueError(f"Error processing HTML: {str(e)}")

def __read_file_content(file_path: str, installed_modules: Dict[str, bool]) -> str:
    """Private function to read content from various file types."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Text files
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # JSON files        
        elif file_extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), indent=2)
        
        # HTML files        
        elif file_extension == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                return __extract_text_from_html(f.read())
        
        # PDF files        
        elif file_extension == '.pdf' and installed_modules.get('pdfplumber'):
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        # Word documents        
        elif file_extension == '.docx' and installed_modules.get('docx'):
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        
        # Spreadsheets        
        elif file_extension in ['.csv', '.xlsx'] and installed_modules.get('pandas'):
            df = pd.read_csv(file_path) if file_extension == '.csv' else pd.read_excel(file_path)
            return f"Data Summary:\n{df.describe().to_string()}\n\nPreview:\n{df.head().to_string()}\n\nShape: {df.shape}"
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        raise ValueError(f"Error reading {file_extension} file: {str(e)}")

def __analyze_content(client: OpenAI, content: str, filename: str, 
                     system_role: str, analysis_task: str, 
                     additional_instructions: str, model: str,
                     max_tokens: int, temperature: float, 
                     chunk_size: int = 4000) -> str:
    """Private function to analyze content using OpenAI API."""
    try:
        content_chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        full_analysis = []

        for i, chunk in enumerate(content_chunks):
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"""
Analysis Task: {analysis_task}
Additional Instructions: {additional_instructions}
File: {filename} (Part {i+1}/{len(content_chunks)})

Content:
{chunk}
                """}
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            full_analysis.append(response.choices[0].message.content)
        
        if len(full_analysis) > 1:
            consolidation_messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": f"""
Please provide a consolidated analysis summary focusing on:
1. Key Points and Insights
2. Overall Patterns and Trends
3. Notable Findings
4. Recommendations
5. Additional Observations

Previous analyses:
{' '.join(full_analysis)}
                """}
            ]
            
            summary_response = client.chat.completions.create(
                model=model,
                messages=consolidation_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return summary_response.choices[0].message.content
        
        return full_analysis[0]
        
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# =====================================
# YFinance Data Functions
# =====================================

def get_stock_info(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/stock_info.json') -> str:
    """Retrieves stock information and saves to JSON."""
    __ensure_directory_exists(output_file_path)
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4)
        return output_file_path
    except Exception as e:
        return f"Error retrieving stock info: {str(e)}"

def get_historical_market_data(ticker_symbol: str = 'MSFT', period: str = '1mo', 
                             output_file_path: str = 'dataspace/historical_data.csv') -> str:
    """Retrieves historical market data and saves to CSV."""
    __ensure_directory_exists(output_file_path)
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period)
        hist.to_csv(output_file_path)
        return output_file_path
    except Exception as e:
        return f"Error retrieving historical data: {str(e)}"

def get_financials(ticker_symbol: str = 'MSFT', output_file_path: str = 'dataspace/financials.xlsx') -> str:
    """Retrieves financial statements and saves to Excel."""
    __ensure_directory_exists(output_file_path)
    try:
        ticker = yf.Ticker(ticker_symbol)
        financials = {
            'Income_Statement': ticker.income_stmt,
            'Balance_Sheet': ticker.balance_sheet,
            'Cashflow': ticker.cashflow,
            'Quarterly_Income_Statement': ticker.quarterly_income_stmt,
            'Quarterly_Balance_Sheet': ticker.quarterly_balance_sheet,
            'Quarterly_Cashflow': ticker.quarterly_cashflow
        }
        with pd.ExcelWriter(output_file_path) as writer:
            for sheet_name, df in financials.items():
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name)
        return output_file_path
    except Exception as e:
        return f"Error retrieving financials: {str(e)}"

# Add other YFinance functions as needed...

# =====================================
# Analysis Functions
# =====================================

def universal_analyzer(
    input_paths: Union[str, List[str]] = 'dataspace/input/',
    output_file: str = 'dataspace/analysis_results.json',
    api_key: str = '',
    system_role: str = "You are a highly skilled analyst with expertise in multiple domains.",
    analysis_task: str = "Analyze this content and provide key insights, patterns, and recommendations.",
    additional_instructions: str = "",
    model: str = "gpt-4o",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    consolidated_analysis: bool = True
) -> Dict:
    """
    Universal content analyzer supporting multiple input files and directories.
    
    Parameters:
    - input_paths: String path or list of paths to analyze
    - output_file: Where to save analysis results
    - api_key: OpenAI API key
    - system_role: Role description for the AI
    - analysis_task: Main analysis task description
    - additional_instructions: Extra analysis instructions
    - model: OpenAI model to use
    - max_tokens: Maximum tokens in response
    - temperature: Response randomness (0-1)
    - consolidated_analysis: Whether to provide a consolidated analysis of all files
    
    Returns:
    - Dict containing analysis results and metadata
    """
    try:
        # Initialize modules and file types
        installed_modules = {
            module: __check_module(module)
            for module in ['docx', 'pdfplumber', 'pandas', 'openpyxl', 'bs4']
        }

        supported_file_types = ['.txt', '.json', '.html']
        if installed_modules.get('docx'):
            supported_file_types.append('.docx')
        if installed_modules.get('pdfplumber'):
            supported_file_types.append('.pdf')
        if installed_modules.get('pandas'):
            supported_file_types.extend(['.csv', '.xlsx'])

        if not api_key:
            raise ValueError("OpenAI API key is required")

        client = OpenAI(api_key=api_key)
        analysis_results = {
            'system_info': {
                'installed_modules': installed_modules,
                'supported_file_types': supported_file_types
            },
            'files': {},
            'consolidated_analysis': None
        }

        def analyze_file(file_path: str) -> Dict:
            """Analyze a single file."""
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in supported_file_types:
                content = __read_file_content(file_path, installed_modules)
                analysis = __analyze_content(
                    client=client,
                    content=content,
                    filename=os.path.basename(file_path),
                    system_role=system_role,
                    analysis_task=analysis_task,
                    additional_instructions=additional_instructions,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return {
                    'file_path': file_path,
                    'analysis': analysis,
                    'file_type': file_extension
                }
            return {
                'file_path': file_path,
                'error': f"Unsupported file type: {file_extension}",
                'file_type': file_extension
            }

        def process_path(path: str) -> List[Dict]:
            """Process a single path (file or directory)."""
            results = []
            if os.path.isfile(path):
                results.append(analyze_file(path))
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        results.append(analyze_file(file_path))
            return results

        # Process all input paths
        all_results = []
        input_paths = [input_paths] if isinstance(input_paths, str) else input_paths
        
        for path in input_paths:
            path_results = process_path(path)
            all_results.extend(path_results)
            
        # Store individual file analyses
        for result in all_results:
            file_name = os.path.basename(result['file_path'])
            analysis_results['files'][file_name] = result

        # Generate consolidated analysis if requested and there are multiple files
        if consolidated_analysis and len(all_results) > 1:
            successful_analyses = [r for r in all_results if 'error' not in r]
            if successful_analyses:
                combined_content = "\n\n".join([
                    f"File: {os.path.basename(r['file_path'])} (Type: {r['file_type']})\n"
                    f"Analysis:\n{r['analysis']}"
                    for r in successful_analyses
                ])
                
                consolidated = __analyze_content(
                    client=client,
                    content=combined_content,
                    filename="All Files",
                    system_role=system_role,
                    analysis_task=f"{analysis_task}\n\nProvide a consolidated analysis of all files.",
                    additional_instructions=f"{additional_instructions}\n\nFocus on connections and patterns across all files.",
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                analysis_results['consolidated_analysis'] = consolidated

        # Save results
        __ensure_directory_exists(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=4, ensure_ascii=False)
            
        return analysis_results

    except Exception as e:
        error_result = {'error': str(e)}
        __ensure_directory_exists(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=4, ensure_ascii=False)
        return error_result

def financial_advisor(
    input_paths: Union[str, List[str]] = 'dataspace/input/',
    output_file: str = 'dataspace/financial_advice.json',
    api_key: str = '',
    investment_context: str = "General investment analysis and recommendations",
    risk_profile: str = "Moderate",
    time_horizon: str = "Long-term (5+ years)",
    analysis_task: str = "Provide comprehensive investment analysis and recommendations",
    model: str = "gpt-4o",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    consolidated_analysis: bool = True
) -> Dict:
    """
    Analyzes multiple financial documents and provides professional investment advice.
    
    Parameters:
    - input_paths: String path or list of paths to financial documents
    - output_file: Where to save the analysis and recommendations
    - api_key: OpenAI API key
    - investment_context: Specific context or goals for the investment analysis
    - risk_profile: Investment risk profile (e.g., Conservative, Moderate, Aggressive)
    - time_horizon: Investment time horizon
    - analysis_task: Specific analysis task or focus
    - model: OpenAI model to use
    - max_tokens: Maximum tokens in response
    - temperature: Response creativity (0-1)
    - consolidated_analysis: Whether to provide a consolidated analysis of all documents
    """
    return universal_analyzer(
        input_paths=input_paths,
        output_file=output_file,
        api_key=api_key,
        system_role="""You are a highly experienced financial advisor with expertise in:
        - Investment analysis and portfolio management
        - Market analysis and trend identification
        - Risk assessment and management
        - Financial planning and strategy development""",
        analysis_task=f"""Investment Context: {investment_context}
Risk Profile: {risk_profile}
Time Horizon: {time_horizon}

{analysis_task}""",
        additional_instructions="""Provide thorough, professional analysis considering:
- The client's risk profile and investment goals
- Current market conditions and trends
- Potential risks and mitigation strategies
- Long-term investment implications""",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        consolidated_analysis=consolidated_analysis
    )
# # Example usage
# if __name__ == "__main__":
#     # Set your OpenAI API key
#     api_key = "your-api-key"
    
#     # Example: Analyze stock information
#     stock_info_path = get_stock_info('AAPL')
    
#     # Example: Analyze the stock information file
#     analysis = financial_advisor(
#         input_path=stock_info_path,
#         api_key=api_key,
#         investment_context="Evaluating Apple Inc. for long-term investment",
#         risk_profile="Moderate",
#         time_horizon="5-10 years"
#     )
    
#     print(analysis)