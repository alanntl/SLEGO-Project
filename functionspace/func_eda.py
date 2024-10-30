import sweetviz as sv
import pandas as pd

# from autoviz import AutoViz_Class
from typing import Union


def generate_sweetviz_report(input_csv: str='dataspace/dataset.csv', 
                             output_html: str = 'dataspace/sweetviz_report.html'):
    """
    Generates a Sweetviz report from a specified CSV file and saves it as an HTML file.

    Parameters:
    input_csv (str): Path to the input CSV dataset.
    output_html (str): Path where the HTML report will be saved, default is 'report.html'.

    Returns:
    None: The function saves the HTML report to the specified path and does not return any value.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)

    # Analyze the dataset
    report = sv.analyze(data)

    # Save the report to an HTML file
    report.show_html(output_html)

    return 'Report has been generated!'



# def autoviz_plot(input_file_path: str = 'dataspace/AirQuality.csv', 
#                  target_variable: Union[str, None] = 'CO(GT)', 
#                  custom_plot_dir: str = 'dataspace',
#                  max_rows_analyzed: int = 150000,
#                  max_cols_analyzed: int = 30,
#                  lowess: bool = False,
#                  header: int = 0,
#                  verbose: int = 2,
#                  sep: str = ''):
#     """
#     Generates visualizations for the dataset using AutoViz.

#     Parameters:
#     input_file_path (str): Path to the input CSV dataset.
#     target_variable (Union[str, None]): Target variable for analysis. If None, no specific target.
#     custom_plot_dir (str): Directory where plots will be saved.
#     max_rows_analyzed (int): Maximum number of rows to analyze.
#     max_cols_analyzed (int): Maximum number of columns to analyze.
#     lowess (bool): Whether to use locally weighted scatterplot smoothing.
#     header (int): Row number to use as the column names.
#     verbose (int): Verbosity level.
#     sep (str): Separator used in the CSV file.

#     Returns:
#     str: Message indicating the completion of the visualization process.
#     """
#     AV = AutoViz_Class()

#     # Perform the AutoViz analysis and generate the plots
#     dft = AV.AutoViz(
#         filename=input_file_path,
#         sep=sep,
#         depVar=target_variable,
#         dfte=None,
#         header=header,
#         verbose=verbose,
#         lowess=lowess,
#         chart_format="html",
#         max_rows_analyzed=max_rows_analyzed,
#         max_cols_analyzed=max_cols_analyzed,
#         save_plot_dir=custom_plot_dir)

#     return "Visualizations have been generated and saved!"
