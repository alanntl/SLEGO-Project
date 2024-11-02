import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional

def __plotly_chart(input_csv: str='dataspace/dataset.csv', 
                 chart_type: str='line',
                 title: str='Chart Title',
                 x_axis: str='Date',
                 y_axes: List[str] = ['Close'],  # Can be a list or a single string
                 y_secondary: Optional[str] = '',  # Optional secondary Y-axis
                 output_html: str='dataspace/plotly_viz.html'):
    """
    Generates a chart using Plotly based on the specified chart type with an optional secondary Y-axis and saves it as an HTML file.

    Parameters:
        input_csv (str): Path to the CSV file containing the data.
        chart_type (str): Type of chart to generate ('line', 'bar', 'scatter').
        x_axis (str): Column name to be used as the x-axis.
        y_axes (list or str): Column name(s) to be used as the primary y-axis. Can be a list for multiple Y-values.
        y_secondary (str, optional): Column name to be used as the secondary y-axis.
        output_html (str): Path where the HTML file will be saved.

    Returns:
        None: The function saves the chart directly to an HTML file and also returns the figure.
    """
    data = pd.read_csv(input_csv)

    # Initialize a Plotly figure
    fig = go.Figure()

    # Process primary y-axes
    if isinstance(y_axes, list):
        for y_axis in y_axes:
            if chart_type == 'line':
                fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axis], name=y_axis, yaxis='y'))
            elif chart_type == 'bar':
                fig.add_trace(go.Bar(x=data[x_axis], y=data[y_axis], name=y_axis, yaxis='y'))
            elif chart_type == 'scatter':
                fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axis], mode='markers', name=y_axis, yaxis='y'))
    else:
        if chart_type == 'line':
            fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axes], name=y_axes, yaxis='y'))
        elif chart_type == 'bar':
            fig.add_trace(go.Bar(x=data[x_axis], y=data[y_axes], name=y_axes, yaxis='y'))
        elif chart_type == 'scatter':
            fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axes], mode='markers', name=y_axes, yaxis='y'))

    # Process secondary y-axis if specified
    if y_secondary==None or y_secondary!='':
        fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_secondary], name=y_secondary, yaxis='y2', marker=dict(color='red')))
        # Create a secondary y-axis configuration
        fig.update_layout(
            yaxis2=dict(
                title=y_secondary,
                overlaying='y',
                side='right'
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=','.join(y_axes) if isinstance(y_axes, list) else y_axes
    )

    # Save the figure as an HTML file and return the figure
    fig.write_html(output_html)
    return fig