import arxiv
import csv
from typing import List, Dict, Optional
import requests
import pandas as pd
import os
from openai import OpenAI

def search_arxiv_papers(search_query: str = 'machine learning', 
                       filename: str = "dataspace/latest_papers.csv", 
                       max_results: int = 5,
                       sort_by: str = "submitted",
                       sort_order: str = "descending") -> pd.DataFrame:
    """
    Searches for papers on arXiv and saves the results to a CSV file.
    """
    sort_options = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submitted": arxiv.SortCriterion.SubmittedDate
    }
    order_options = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending
    }

    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=sort_options.get(sort_by, arxiv.SortCriterion.SubmittedDate),
        sort_order=order_options.get(sort_order, arxiv.SortOrder.Descending)
    )
    
    try:
        results = list(search.results())
        
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
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        return df
    
    except Exception as e:
        print(f"Error searching arXiv: {str(e)}")
        return pd.DataFrame()

def download_papers_from_arxiv_csv(filename: str = "dataspace/latest_papers.csv", 
                                 download_folder: str = "dataspace/papers/",
                                 url_col: str = "entry_id",
                                 title_col: str = "title") -> str:
    """
    Download papers from arXiv based on a CSV file.
    """
    try:
        os.makedirs(download_folder, exist_ok=True)
        df = pd.read_csv(filename)
        
        for index, row in df.iterrows():
            try:
                arxiv_url = row[url_col]
                title = row[title_col].replace('/', '_').replace(':', '_')
                arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0]
                
                download_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                file_path = os.path.join(download_folder, f"{title}.pdf")
                
                response = requests.get(download_url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download {title} with ID {arxiv_id}")
                    
            except Exception as e:
                print(f"Error downloading paper {row[title_col]}: {str(e)}")
                continue
                
        return 'Download finished, please check your folder!'
    
    except Exception as e:
        return f"Error processing downloads: {str(e)}"

def analyze_research_trends(
    filename: str = "dataspace/latest_papers.csv", 
    output_filename: str = "dataspace/research_trends.txt",
    api_key: str = None,
    analysis_task: str = "Analyze research trends in the provided papers.",
    additional_instructions: str = "",
    model: str = "gpt-4o",
    max_tokens: int = 1500,
    temperature: float = 0.7,
    system_role: str = "You are an expert data analyst specializing in research trends.",
    chunk_size: int = 1500
) -> str:
    """
    Analyzes research trends from arXiv papers using OpenAI's GPT model.
    """
    try:
        df = pd.read_csv(filename)
        titles_and_summaries = df['title'] + '. ' + df['summary']
        combined_text = '\n\n'.join(titles_and_summaries)
        
        analysis = __analyze_content(
            api_key=api_key,
            content=combined_text,
            filename=filename,
            system_role=system_role,
            analysis_task=analysis_task,
            additional_instructions=additional_instructions,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=chunk_size
        )
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write(analysis)
            
        return analysis
    
    except Exception as e:
        return f"Error analyzing research trends: {str(e)}"

def __analyze_content(
    api_key: str,
    content: str,
    filename: str,
    system_role: str,
    analysis_task: str,
    additional_instructions: str,
    model: str,
    max_tokens: int,
    temperature: float,
    chunk_size: int = 4000
) -> str:
    """Private function to analyze content using OpenAI API."""
    try:
        if not api_key:
            return "Error: OpenAI API key is required"
            
        client = OpenAI(api_key=api_key)
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

Previous Analyses:
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
        return f"Error during content analysis: {str(e)}"