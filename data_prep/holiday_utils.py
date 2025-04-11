# Retrieves holidays and formats them into a compatible DataFrame

import pandas as pd
import holidays

def get_holidays(years, country="DE"):
    """
    Retrieve holidays for a specific country and format them into a DataFrame.
    Parameters
    ----------
    years : List[int]
        List of years for which to retrieve holidays.
    country : str
        Country code (e.g., "DE" for Germany, "CH" for Switzerland, "GB" for the UK).
    Returns
    -------
    pd.DataFrame
        Holidays dataframe with columns 'ds' and 'holiday'.
    """
    holiday_list = holidays.CountryHoliday(country, years=years)
    holidays_df = pd.DataFrame(list(holiday_list.items()), columns=["ds", "holiday"])
    holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])
    return holidays_df