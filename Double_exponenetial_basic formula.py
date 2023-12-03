import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import matplotlib.dates as mdates
from sklearn import metrics

engine = create_engine('postgresql://stockeradminsimec:stock_admin_#146@localhost/stockerhubdb')

# Specify the schema and table name
schema_name = 'public'
table_name = 'company_price_companyprice'

# Specify the target name to filter
target_instr_code = 'GLOBALINS'

try:
    engine_sc = engine
    selected_columns = ['mkt_info_date', 'closing_price']
    # Construct the SQL query to select data from the specified table and filter by name
    query = f"SELECT {', '.join(selected_columns)} FROM {table_name} WHERE \"Instr_Code\" = '{target_instr_code}'"
    # print(query)
    # Execute the query and load the results into a DataFrame
    df = pd.read_sql(query, engine_sc)
    closing_prices = df['closing_price']


    # Function to perform Double Exponential Smoothing
    def double_exponential_smoothing(series, alpha, beta):
        n = len(series)
        forecast = np.zeros(n)

        # Initial values
        level = series[0]
        trend = series[1] - series[0]

        forecast[0] = series[0]
        forecast[1] = series[1]

        # Double Exponential Smoothing
        for t in range(2, n):
            forecast[t] = level + trend
            level = alpha * series[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - forecast[t - 1]) + (1 - beta) * trend

        return forecast


    # Set your alpha and beta values
    alpha = 0.2
    beta = 0.2

    forecast = double_exponential_smoothing(closing_prices, alpha, beta)

    print(forecast)
    # Print the resulting DataFrame
    # print(df)

    # Print the result of Residual Differences
    residual_difference = pd.DataFrame({'Actual': closing_prices, 'Predicted': forecast, 'Loss': abs(closing_prices - forecast)})
    print(residual_difference)

    # check accuracy
    print('Accuracy:', metrics.r2_score(closing_prices, forecast))

    # Plotting the results
    df['mkt_info_date'] = pd.to_datetime(df['mkt_info_date'])
    df = df.sort_values('mkt_info_date')
    # df['Date']= df['Date'].sort_values()

    plt.plot(df['mkt_info_date'], closing_prices, label='Actual Closing Prices')
    plt.plot(df['mkt_info_date'], forecast, label='Double Exponential Smoothing Forecast')
    plt.legend()
    plt.title('Double Exponential Smoothing Forecasting')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate(rotation=60)
    plt.xlim(df['mkt_info_date'].min(), df['mkt_info_date'].max())
    plt.text(0.95, 0.9, f'Alpha: {alpha}', transform=plt.gca().transAxes, ha='right', va='center')
    plt.text(0.95, 0.85, f'Beta: {beta}', transform=plt.gca().transAxes, ha='right', va='center')
    plt.show()





except Exception as e:
    print(f"Error: {e}")
finally:
    # Close the connection if it was successful
    if engine:
        engine.dispose()