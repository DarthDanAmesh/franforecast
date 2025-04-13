# Functions for working with dates (e.g., formatting, resampling)
import pandas as pd

def format_date_column(df, date_col, freq="M"):
    """
    Format the date column to datetime and resample the data.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.
    freq : str
        Resampling frequency (e.g., "D" for daily, "W" for weekly, "M" for monthly).
    Returns
    -------
    pd.DataFrame
        Dataframe with formatted date column and resampled data.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df = df.resample(freq).sum()  # Aggregate by summing up values
    return df

def add_time_features(df, date_col="Date"):
    """
    Add time-based features to the DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.
    Returns
    -------
    pd.DataFrame
        Dataframe with additional time-based features.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract time-based features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday  # Monday=0, Sunday=6
    df["is_weekend"] = df[date_col].dt.weekday.isin([5, 6]).astype(int)  # 1 if weekend, else 0
    
    return df


from datetime import datetime, timedelta
import pandas as pd

def generate_dates(start_date, end_date, aggregation_level, num_periods=None):
    """
    Generate date range based on aggregation level
    
    Parameters:
    -----------
    start_date : datetime or str
        Start date of the time series
    end_date : datetime or str
        End date of the time series
    aggregation_level : str
        One of: "Daily", "Weekly", "Monthly"
    num_periods : int, optional
        If provided, generates this many periods from start_date
        
    Returns:
    --------
    pd.DatetimeIndex
        Array of dates at the specified frequency
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    freq_map = {
        "Daily": "D",
        "Weekly": "W-MON",  # Weekly starting Monday
        "Monthly": "MS"  # Month start
    }
    
    if num_periods:
        return pd.date_range(
            start=start_date,
            periods=num_periods,
            freq=freq_map.get(aggregation_level, "D")
        )
    else:
        return pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq_map.get(aggregation_level, "D")
        )