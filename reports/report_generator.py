# Generates summary reports (HTML, PDF, or Streamlit-based)

import pandas as pd
import streamlit as st
from datetime import datetime

def generate_html_report(metrics_df, filename="report.html"):
    """
    Generate an HTML report summarizing the evaluation metrics.
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing evaluation metrics.
    filename : str
        Name of the output HTML file.
    """
    html_content = f"""
    <html>
    <head><title>Forecasting Report - {datetime.now().strftime('%Y-%m-%d')}</title></head>
    <body>
        <h1>Forecasting Report</h1>
        <p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>
        <h2>Evaluation Metrics</h2>
        {metrics_df.to_html(index=False)}  <!-- Remove index for cleaner output -->
    </body>
    </html>
    """
    with open(filename, "w") as f:
        f.write(html_content)
    st.success(f"HTML report generated: {filename}")

def generate_streamlit_report(metrics_df, forecasts_df):
    """
    Generate an interactive report in Streamlit.
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing evaluation metrics.
    forecasts_df : pd.DataFrame
        Dataframe containing forecasted values.
    """
    st.header("Forecasting Report")
    st.subheader("Evaluation Metrics")
    st.dataframe(metrics_df)

    st.subheader("Forecasted Values")
    st.dataframe(forecasts_df)

def export_to_pdf(metrics_df, filename="report.pdf"):
    """
    Export the report to a PDF file (requires additional libraries like WeasyPrint).
    Parameters
    ----------
    metrics_df : pd.DataFrame
        Dataframe containing evaluation metrics.
    filename : str
        Name of the output PDF file.
    """
    try:
        from weasyprint import HTML
        html_content = f"""
        <html>
        <body>
            <h1>Forecasting Report</h1>
            <p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>
            <h2>Evaluation Metrics</h2>
            {metrics_df.to_html()}
        </body>
        </html>
        """
        HTML(string=html_content).write_pdf(filename)
        st.success(f"PDF report generated: {filename}")
    except ImportError:
        st.error("WeasyPrint is required to generate PDF reports. Install it using `pip install weasyprint`.")