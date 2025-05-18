import pandas as pd

def get_period(year_month, month_range):
    date_range = []
    for i in range(month_range):
        one_date = pd.to_datetime(year_month, format="%Y-%m-%d") - pd.DateOffset(months=i+1)
        formatted_date_string = "'"+one_date.strftime("%d.%m.%Y")+"'"
        date_range.append(formatted_date_string)
    
    return ','.join(date_range)