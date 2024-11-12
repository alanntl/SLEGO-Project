import PyPDF2
import os
import sys
import argparse
from datetime import datetime
from typing import Tuple

def pdf_to_text(
    input_pdf: str = "dataspace/input.pdf",
    output_txt: str = "dataspace/output.txt",
    encoding: str = 'utf-8',
    verbose: bool = False,
    create_timestamp: bool = True
) -> Tuple[str, str]:
    """
    Convert a PDF file to text with configurable options.
    
    Args:
        input_pdf (str): Path to the input PDF file. Defaults to "input.pdf"
        output_txt (str): Path to save the output text file. Defaults to "output.txt"
        encoding (str): Character encoding for output file. Defaults to 'utf-8'
        verbose (bool): If True, prints progress information. Defaults to False
        create_timestamp (bool): If True, adds timestamp to output filename. Defaults to True
    
    Returns:
        Tuple[str, str]: (Extracted text content, Path to output file)
        
    Raises:
        FileNotFoundError: If input PDF file doesn't exist
        PermissionError: If unable to read input file or write output file
    """
    try:
        if verbose:
            print(f"Processing PDF file: {input_pdf}")
            
        # Open and read PDF file
        with open(input_pdf, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            if verbose:
                print(f"Number of pages: {num_pages}")
            
            # Extract text from all pages
            text = ""
            for page_num in range(num_pages):
                if verbose:
                    print(f"Processing page {page_num + 1}/{num_pages}")
                text += pdf_reader.pages[page_num].extract_text()
        
        # Prepare output filename
        output_dir = os.path.dirname(output_txt) or '.'
        output_basename = os.path.basename(output_txt)
        output_name, output_ext = os.path.splitext(output_basename)
        
        # Add timestamp if requested
        if create_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output = os.path.join(output_dir, f"{output_name}_{timestamp}{output_ext or '.txt'}")
        else:
            final_output = os.path.join(output_dir, f"{output_name}{output_ext or '.txt'}")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
            
        # Save extracted text to file
        with open(final_output, 'w', encoding=encoding) as txt_file:
            txt_file.write(text)
            
        if verbose:
            print(f"Text successfully saved to: {final_output}")
            
        return text, final_output
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input PDF file '{input_pdf}' not found")
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing '{input_pdf}' or '{final_output}'")
    except UnicodeEncodeError:
        raise UnicodeEncodeError(f"Error encoding text with {encoding}. Try a different encoding.")
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")