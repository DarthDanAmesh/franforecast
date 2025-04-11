import pandas as pd
import streamlit as st

def load_data(file):
    """Improved data loading with consistent column naming"""
    try:
        df = pd.read_csv(file)
        
        # Normalize all column names (strip whitespace, capitalize first letter)
        df.columns = [col.strip().capitalize() for col in df.columns]
        
        # Standardize date column name to 'Date'
        date_col = None
        for col in df.columns:
            lower_col = col.lower()
            if lower_col in ['date', 'datetime', 'time']:
                date_col = col
                break
                
        if date_col:
            # Rename to standard 'Date' column
            if date_col != 'Date':
                df = df.rename(columns={date_col: 'Date'})
            # Convert to datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        elif all(c in df.columns for c in ['Year', 'Month', 'Day']):
            df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        else:
            raise ValueError("No valid date columns found")
            
        # Check if we have any valid dates
        if df['Date'].isna().all():
            raise ValueError("Could not parse any valid dates from the data")
            
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Standardize other common columns
        column_mapping = {
            'Product_id': 'Item_code',
            'Quantity': 'target',  # Use lowercase "target"
            'Product_description': 'Description'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() 
                               if k in df.columns and v not in df.columns})
        
        st.success("Data loaded successfully with standardized column names")
        st.write("Final standardized columns:", df.columns.tolist())
        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preview_data(df, n=5):
    """
    Display a preview of the data in the Streamlit app.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    n : int
        Number of rows to display.
    """
    if df is not None:
        st.write("Preview of the data:")
        st.dataframe(df.head(n))