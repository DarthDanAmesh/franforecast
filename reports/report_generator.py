# Generates summary reports (HTML, PDF, or Streamlit-based)
import plotly.express as px
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

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



def generate_streamlit_report(metrics, forecasts, test_df):
    """
    Enhanced reporting with proper metrics handling using Plotly.
    """
    st.subheader("Detailed Forecast Analysis")
    
    # 1. Metrics display
    with st.expander("Model Metrics"):
        # Convert metrics dict to DataFrame
        metrics_df = pd.DataFrame([metrics])
        st.table(metrics_df)
    
    # 2. Error distribution
    with st.expander("Error Analysis"):
        try:
            # Ensure forecasts is numpy array
            forecasts = np.asarray(forecasts)
            
            # Get corresponding actual values
            actual_values = test_df["target"].values[-len(forecasts):]
            
            # Calculate errors
            errors = actual_values - forecasts
            
            # Plot histogram if we have data
            if len(errors) > 0:
                # Create a Plotly histogram
                fig = px.histogram(
                    x=errors,
                    nbins=30,
                    title="Prediction Error Distribution",
                    labels={"x": "Error", "y": "Frequency"},
                    color_discrete_sequence=["#2ca02c"]
                )
                
                # Update layout for better appearance
                fig.update_layout(
                    xaxis_title="Error",
                    yaxis_title="Frequency",
                    margin=dict(t=50),
                    height=400
                )
                
                # Show the histogram
                st.plotly_chart(fig, use_container_width=True)
                
                # Show error statistics
                error_stats = {
                    "Mean Error": np.mean(errors),
                    "Std Dev": np.std(errors),
                    "Min Error": np.min(errors),
                    "Max Error": np.max(errors)
                }
                st.table(pd.DataFrame([error_stats]))
            else:
                st.warning("No error data available for visualization")
                
        except Exception as e:
            st.warning(f"Could not show error analysis: {str(e)}")
            st.write("Debug info:")
            st.write(f"Actual values shape: {test_df['target'].shape}")
            st.write(f"Forecasts shape: {forecasts.shape}")


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