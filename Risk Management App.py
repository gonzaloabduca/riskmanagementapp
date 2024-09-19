import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
import pandas_datareader as pdr

# Define the function to calculate Sharpe Ratio
def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

# Define the function to calculate Sortino Ratio
def sortino_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    std_neg = return_series[return_series < 0].std() * np.sqrt(N)
    return mean / std_neg

# Define the function to calculate Maximum Drawdown
def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

# Define the function to calculate CVaR
def calculate_cvar(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = sorted_returns[:index].mean()
    return cvar

# Set the parameters
N = 252  # Number of trading days in a year

# Compute MACD function
def compute_macd(df, short_window=15, long_window=60, signal_window=9):
    df['EMA12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD Histogram'] = df['MACD'] - df['Signal Line']
    return df

# Function to compute RSI
def compute_rsi(df, window=20):
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    RS = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + RS))
    df['RSI'].fillna(50, inplace=True)
    return df

# Generate MACD and RSI-based signals
def generate_signals(df):
    # Calculate the mean and standard deviation for the MACD Histogram
    years = 252 * 2
    df['MACD Rolling Mean'] = df['MACD Histogram'].rolling(window=years).mean()
    df['MACD Rolling Std'] = df['MACD Histogram'].rolling(window=years).std() / 2

    # Assign MACD-based signals based on standard deviation
    df['MACD Signal'] = 'Neutral'  # Default to Neutral
    df.loc[df['MACD Histogram'] > (df['MACD Rolling Mean'] + df['MACD Rolling Std']), 'MACD Signal'] = 'Bullish'
    df.loc[df['MACD Histogram'] < (df['MACD Rolling Mean'] - df['MACD Rolling Std']), 'MACD Signal'] = 'Bearish'
    
    # Assign RSI-based signals
    df['RSI Signal'] = 'Neutral'  # Default to Neutral
    df.loc[df['RSI'] > 60, 'RSI Signal'] = 'Bullish'
    df.loc[df['RSI'] < 40, 'RSI Signal'] = 'Bearish'
    
    return df

# Function to calculate trend indicator
def calculate_trend_indicator(df, window=252, std_multiplier=0.25):
    # Calculate the rolling mean return and the rolling average price
    df['Return'] = df['Adj Close'].pct_change()  # Calculate daily return
    df['Rolling Mean Return'] = df['Return'].rolling(window=window).mean()  # Rolling mean of returns
    df['Rolling Avg Price'] = df['Adj Close'].rolling(window=window).mean()  # Rolling average price
    
    # Calculate the upper and lower bands with +std_multiplier and -std_multiplier
    df['Rolling Std'] = df['Adj Close'].rolling(window=window).std()
    df['Upper Band'] = ((1 + df['Rolling Mean Return']) * df['Rolling Avg Price']) + std_multiplier * df['Rolling Std']
    df['Lower Band'] = ((1 + df['Rolling Mean Return']) * df['Rolling Avg Price']) - std_multiplier * df['Rolling Std']
    
    # Define the trend based on conditions
    def determine_trend(row):
        if row['Adj Close'] > row['Upper Band']:
            return 'Bullish Trend'
        elif row['Adj Close'] < row['Lower Band']:
            return 'Bearish Trend'
        else:
            return 'Neutral'
    
    df['Trend'] = df.apply(determine_trend, axis=1)
    return df

# Generate Momentum Signals based on the combination of MACD and RSI signals
def generate_momentum_signals(df):
    df['Momentum Signal'] = 'Neutral'
    df.loc[(df['MACD Signal'] == 'Bullish') & (df['RSI Signal'] == 'Bullish'), 'Momentum Signal'] = 'Strong Bullish Momentum'
    df.loc[(df['MACD Signal'] == 'Bullish') & (df['RSI Signal'] != 'Bullish'), 'Momentum Signal'] = 'Weak Bullish Momentum'
    df.loc[(df['RSI Signal'] == 'Bullish') & (df['MACD Signal'] != 'Bullish'), 'Momentum Signal'] = 'Weak Bullish Momentum'
    df.loc[(df['MACD Signal'] == 'Neutral') & (df['RSI Signal'] == 'Neutral'), 'Momentum Signal'] = 'Neutral'
    df.loc[(df['MACD Signal'] == 'Bearish') & (df['RSI Signal'] != 'Bearish'), 'Momentum Signal'] = 'Weak Bearish Momentum'
    df.loc[(df['RSI Signal'] == 'Bearish') & (df['MACD Signal'] != 'Bearish'), 'Momentum Signal'] = 'Weak Bearish Momentum'
    df.loc[(df['MACD Signal'] == 'Bearish') & (df['RSI Signal'] == 'Bearish'), 'Momentum Signal'] = 'Strong Bearish Momentum'
    return df

# Main function to compute the data and signals
def calculate_signals(ticker):
    end = datetime.now()
    start = end - timedelta(days=365*10)
    
    # Download data
    df = yf.download(ticker, start=start, end=end)
    
    # Apply MACD, RSI, and Trend calculations
    df = compute_macd(df)
    df = compute_rsi(df)
    df = generate_signals(df)
    df = generate_momentum_signals(df)
    
    # Apply the trend indicator and merge it with the signals
    df = calculate_trend_indicator(df)
    
    return df

# Title of the app
st.title("Risk Management Trading App")

# Sidebar for ticker symbol input
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value='AAPL', max_chars=10)

# Fetch company information
company_info = yf.Ticker(ticker).info
company_name = company_info.get("shortName", "Unknown Company")

# Date range buttons
time_ranges = {
    "1 Month": timedelta(days=30),
    "3 Months": timedelta(days=90),
    "6 Months": timedelta(days=180),
    "1 Year": timedelta(days=365),
    "2 Years": timedelta(days=730),
    "5 Years": timedelta(days=1825)
}


# Default time range
selected_range = "1 Year"

# Horizontal layout for buttons under the chart
st.markdown("<br>", unsafe_allow_html=True)  # Adding a line break for spacing
cols = st.columns(len(time_ranges))

# Create buttons for each time range
for i, key in enumerate(time_ranges.keys()):
    if cols[i].button(key):
        selected_range = key

# Calculate the start date based on the selected time range
end_date = datetime.now()
start_date = end_date - time_ranges[selected_range]

end = datetime.now()
start = end - timedelta(days=365*10)

cagr_end = datetime.now()
cagr_start = cagr_end - timedelta(days=365*10)

# Fetch and calculate signals
signal_data = calculate_signals(ticker)


# Fetching data
data = yf.download(ticker, start=start_date, end=end_date)

data2 = yf.download(ticker, start=start, end=end)


# Fetching monthly data
monthly_data_2 = yf.download(ticker, start=cagr_start, end=cagr_end, interval='1mo')

rf_data = pdr.get_data_fred('DGS1MO', start=start_date, end=end_date).interpolate()
rf = rf_data.iloc[-1, 0] / 100  # Last available 1-Month Treasury rate as risk-free rate

# Rolling 12-Month CAGR Calculation
rolling_12m_cagr = (monthly_data_2['Adj Close'].pct_change(12) + 1) ** (1 / 1) - 1
rolling_12m_cagr = rolling_12m_cagr * 100  # Convert to percentage

# Create a bar chart for the last 3 years of Rolling 12-Month CAGR
fig6 = go.Figure()

# Add bars to the chart with conditional coloring
fig6.add_trace(go.Bar(
    x=rolling_12m_cagr.index,
    y=rolling_12m_cagr,
    marker_color=['green' if cagr >= 0 else 'red' for cagr in rolling_12m_cagr],
))

# Update layout for the bar chart
fig6.update_layout(title='Rolling 12-Month CAGR (Last 3 Years)',
                   xaxis_title='Date',
                   yaxis_title='CAGR (%)',
                   showlegend=False)


# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])

close_price = data2['Adj Close'].iloc[-1]

# Update layout for better visualization
fig.update_layout(title=f'Candlestick Chart for {company_name} ({ticker}) - {selected_range} - Last Price: $ {close_price:.2f}',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=False,
                  yaxis_type='log')

# Display the chart in the Streamlit app
st.plotly_chart(fig)
st.plotly_chart(fig6)

# Show Signal Data in Streamlit
st.markdown(f"### Trading Signals for {ticker}")
st.table(signal_data[['MACD Signal', 'RSI Signal', 'Momentum Signal', 'Trend']].tail(1).T)  # Show last few rows of signal data


# Calculate performance metrics for the ticker and benchmarks
benchmarks = ['SPY', 'QQQ']
tickers = benchmarks + [ticker]

# Download the data for benchmarks and ticker
data_benchmarks = yf.download(tickers, start=start, end=end)['Adj Close']

# Calculate daily returns
returns = data_benchmarks.pct_change().dropna()

# Calculate performance metrics
metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'CVar', 'Maximum Drawdown', 'Kurtosis', 'Skewness']
performance_df = pd.DataFrame(index=metrics, columns=tickers)

# Total Return
performance_df.loc['Total Return'] = (data_benchmarks.iloc[-1] / data_benchmarks.iloc[0] - 1) * 100

# Annualized Return
performance_df.loc['Annualized Return'] = ((1 + performance_df.loc['Total Return'] / 100) ** (255 / len(data_benchmarks)) - 1) * 100

# Standard Deviation
performance_df.loc['Standard Deviation'] = returns.std() * np.sqrt(255) * 100

# Sharpe Ratio
performance_df.loc['Sharpe Ratio'] = returns.apply(lambda x: sharpe_ratio(x, 255, 0.01))

# Sortino Ratio
performance_df.loc['Sortino Ratio'] = returns.apply(lambda x: sortino_ratio(x, 255, 0.01))

# Calmar Ratio
max_drawdowns = returns.apply(max_drawdown)
performance_df.loc['Calmar Ratio'] = returns.mean() * 255 / abs(max_drawdowns)

# CVaR
performance_df.loc['CVar'] = returns.apply(calculate_cvar) * 100

# Maximum Drawdown
performance_df.loc['Maximum Drawdown'] = max_drawdowns * 100

# Kurtosis
performance_df.loc['Kurtosis'] = returns.kurtosis()

# Skewness
performance_df.loc['Skewness'] = returns.skew()

# Format as percentages with 2 decimal places for specific metrics
percentage_metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'CVar', 'Maximum Drawdown']
performance_df.loc[percentage_metrics] = performance_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2f}%")

# Format other metrics as floats with 2 decimal places
float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Kurtosis', 'Skewness']  # Corrected 'Kurtrosis' to 'Kurtosis'
performance_df.loc[float_metrics] = performance_df.loc[float_metrics].applymap(lambda x: f"{x:.2f}")


# Display the DataFrame in the app
st.markdown("### Performance Metrics Comparison")
st.table(performance_df)

# Calculate performance for specified periods
def calculate_performance(data, period_days):
    if len(data) < period_days:
        return None
    return (data['Close'][-1] - data['Close'][-period_days]) / data['Close'][-period_days] * 100

performances = {
    "1 Month": calculate_performance(data, 30),
    "3 Months": calculate_performance(data, 90),
    "6 Months": calculate_performance(data, 180),
    "1 Year": calculate_performance(data, 365)
}

# Display performances
st.markdown("### Performance")
performance_cols = st.columns(len(performances))
for i, (label, performance) in enumerate(performances.items()):
    if performance is not None:
        color = "green" if performance > 0 else "red"
        performance_cols[i].markdown(f"<span style='color:{color}'>{label}: {performance:.2f}%</span>", unsafe_allow_html=True)
    else:
        performance_cols[i].markdown(f"{label}: N/A")

# Fetching monthly data
monthly_data = yf.download(ticker, start=start, end=end, interval='1mo')

# Calculate Average Positive and Negative Monthly Returns
monthly_returns = monthly_data['Adj Close'].pct_change().dropna()
positive_monthly_returns = monthly_returns[monthly_returns > 0]
negative_monthly_returns = monthly_returns[monthly_returns < 0]

average_positive_monthly_return = positive_monthly_returns.mean() * 100
average_negative_monthly_return = negative_monthly_returns.mean() * 100

# Calculate Percentage of Positive and Negative Months
total_months = len(monthly_returns)
positive_months = len(positive_monthly_returns)
negative_months = len(negative_monthly_returns)

percentage_positive_months = (positive_months / total_months) * 100
percentage_negative_months = (negative_months / total_months) * 100

# Calculate Risk Reward Profile
if average_negative_monthly_return != 0:
    risk_reward_ratio = abs(average_positive_monthly_return / average_negative_monthly_return)
else:
    risk_reward_ratio = np.nan  # Avoid division by zero


# Calculate the 75th and 25th percentiles
percentile_75 = monthly_returns.quantile(0.95)
percentile_25 = monthly_returns.quantile(0.05)

# Calculate the average return above the 75th percentile
average_above_75th = monthly_returns[monthly_returns > percentile_75].mean()

# Calculate the average return below the 25th percentile
average_below_25th = monthly_returns[monthly_returns < percentile_25].mean()

# Calculate the risk-reward ratio
if average_below_25th != 0:
    risk_reward_ratio_percentiles = average_above_75th / abs(average_below_25th)
else:
    risk_reward_ratio_percentiles = np.nan  # Avoid division by zero

# Calculate Mathematical Expectation
average_win = average_positive_monthly_return / 100
average_loss = abs(average_negative_monthly_return / 100)
winning_percentage = percentage_positive_months / 100
losing_percentage = percentage_negative_months / 100

expectation = (average_win * winning_percentage) - (average_loss * losing_percentage)

# Best Month and Worst Month
best_month = monthly_returns.max() * 100
worst_month = monthly_returns.min() * 100

# Define metrics
monthly_performance_metrics = ['Average Positive Monthly Return', 'Average Negative Monthly Return', 'Percentage of Positive Months', 
                               'Percentage of Negative Months', 'Best Month', 'Worst Month', 'Risk Reward Profile', 
                               'CVar Risk-Reward Ratio', 'Mathematical Expectation (E(x))']

# Initialize the DataFrame
monthly_performance_metrics_df = pd.DataFrame(index=monthly_performance_metrics, columns=[ticker])

# Populate the DataFrame
monthly_performance_metrics_df.loc['Average Positive Monthly Return'] = average_positive_monthly_return
monthly_performance_metrics_df.loc['Average Negative Monthly Return'] = average_negative_monthly_return
monthly_performance_metrics_df.loc['Percentage of Positive Months'] = percentage_positive_months
monthly_performance_metrics_df.loc['Percentage of Negative Months'] = percentage_negative_months
monthly_performance_metrics_df.loc['Best Month'] = best_month
monthly_performance_metrics_df.loc['Worst Month'] = worst_month
monthly_performance_metrics_df.loc['Risk Reward Profile'] = risk_reward_ratio
monthly_performance_metrics_df.loc['CVar Risk-Reward Ratio'] = risk_reward_ratio_percentiles
monthly_performance_metrics_df.loc['Mathematical Expectation (E(x))'] = expectation

# Format percentage metrics
percentage_metrics = ['Average Positive Monthly Return', 'Average Negative Monthly Return', 'Percentage of Positive Months', 
                      'Percentage of Negative Months', 'Best Month', 'Worst Month']
monthly_performance_metrics_df.loc[percentage_metrics] = monthly_performance_metrics_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2f}%")

# Format other metrics as floats with 2 decimal places
monthly_float_metrics = ['Risk Reward Profile', 'CVar Risk-Reward Ratio', 'Mathematical Expectation (E(x))']
monthly_performance_metrics_df.loc[monthly_float_metrics] = monthly_performance_metrics_df.loc[monthly_float_metrics].applymap(lambda x: f"{x:.2f}")

# Display the DataFrame in the app
st.markdown("### Monthly Performance Metrics")
st.table(monthly_performance_metrics_df)

# # Display monthly performance metrics
# st.markdown("### Monthly Performance Metrics")
# st.markdown(f"**Average Positive Monthly Return**: {average_positive_monthly_return:.2f}%")
# st.markdown(f"**Average Negative Monthly Return**: {average_negative_monthly_return:.2f}%")
# st.markdown(f"**Percentage of Positive Months**: {percentage_positive_months:.2f}%")
# st.markdown(f"**Percentage of Negative Months**: {percentage_negative_months:.2f}%")
# st.markdown(f"**Best Month**: {best_month:.2f}%")
# st.markdown(f"**Worst Month**: {worst_month:.2f}%")
# st.markdown(f"**Risk Reward Profile**: {risk_reward_ratio:.2f}x")
# st.markdown(f"**CVar Risk-Reward Ratio**: {risk_reward_ratio_percentiles:.2f}")
# st.markdown(f"**Mathematical Expectation (E(x))**: {expectation:.4f}")

data_spy = yf.download(tickers='SPY', start=start, end=end)
data_qqq = yf.download(tickers='QQQ', start=start, end=end)

data_spy['DailyReturn'] = data_spy['Adj Close'].pct_change().dropna()
data_qqq['DailyReturn'] = data_qqq['Adj Close'].pct_change().dropna()
data['DailyReturn'] = data['Adj Close'].pct_change().dropna()

# Calculate the correlation with SPY and QQQ
correlation_spy = data['DailyReturn'].corr(data_spy['DailyReturn'])
correlation_qqq = data['DailyReturn'].corr(data_qqq['DailyReturn'])

# Display the correlation with SPY and QQQ
st.markdown("### Correlation with Market Indices")
st.markdown(f"**Correlation with S&P 500**: {correlation_spy:.1%}")
st.markdown(f"**Correlation with Nasdaq 100**: {correlation_qqq:.1%}")

# Calculate daily returns
data2['DailyReturn'] = data2['Close'].pct_change()

# Calculate average return by day of the week
data2['DayOfWeek'] = data2.index.dayofweek
average_return_by_day_of_week = data2.groupby('DayOfWeek')['DailyReturn'].mean() * 100
average_return_by_day_of_week.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Calculate average return by day of the month
data2['DayOfMonth'] = data2.index.day
average_return_by_day_of_month = data2.groupby('DayOfMonth')['DailyReturn'].mean() * 100

# Create bar chart for average return by day of the week
fig4 = go.Figure(data=[go.Bar(x=average_return_by_day_of_week.index, y=average_return_by_day_of_week)])
fig4.update_layout(title='Average Return by Day of the Week',
                   xaxis_title='Day of the Week',
                   yaxis_title='Average Return (%)')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig4)

# Create bar chart for average return by day of the month
fig5 = go.Figure(data=[go.Bar(x=average_return_by_day_of_month.index, y=average_return_by_day_of_month)])
fig5.update_layout(title='Average Return by Day of the Month',
                   xaxis_title='Day of the Month',
                   yaxis_title='Average Return (%)')

# Display the bar chart in the Streamlit app
st.plotly_chart(fig5)

# Calculate Historical Relative Volume
data['3MonthAvgVolume'] = data['Volume'].rolling(window=90).mean()
data['RelativeVolume'] = data['Volume'] / data['3MonthAvgVolume']

# Create relative volume chart
fig2 = go.Figure(data=[go.Bar(x=data.index, y=data['RelativeVolume'])])

# Update layout for better visualization
fig2.update_layout(title='Historical Relative Volume',
                   xaxis_title='Date',
                   yaxis_title='Relative Volume')

# Display the relative volume chart in the Streamlit app
st.plotly_chart(fig2)

# Calculate volume change for specified periods
def calculate_volume_change(data, period_days):
    if len(data) < period_days:
        return None
    return (data['Volume'][-1] - data['Volume'][-period_days]) / data['Volume'][-period_days] * 100

volume_changes = {
    "1 Month": calculate_volume_change(data, 30),
    "3 Months": calculate_volume_change(data, 90),
    "6 Months": calculate_volume_change(data, 180),
    "1 Year": calculate_volume_change(data, 365)
}

# Display volume changes
st.markdown("### Volume Changes")
volume_change_cols = st.columns(len(volume_changes))
for i, (label, change) in enumerate(volume_changes.items()):
    if change is not None:
        color = "green" if change > 0 else "red"
        volume_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
    else:
        volume_change_cols[i].markdown(f"{label}: N/A")

# Calculate 30-day annualized volatility
data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
data['30DayVolatility'] = data['LogReturns'].rolling(window=30).std() * np.sqrt(252)

# Create 30-day annualized volatility chart
fig3 = go.Figure(data=[go.Scatter(x=data.index, y=data['30DayVolatility'], mode='lines')])

# Update layout for better visualization
fig3.update_layout(title='30-Day Annualized Volatility',
                   xaxis_title='Date',
                   yaxis_title='Annualized Volatility')

# Display the 30-day annualized volatility chart in the Streamlit app
st.plotly_chart(fig3)

# Calculate current 30-day annualized volatility and its changes
current_volatility = data['30DayVolatility'].iloc[-1]*100

def calculate_volatility_change(data, period_days):
    if len(data) < period_days:
        return None
    return (data['30DayVolatility'].iloc[-1] - data['30DayVolatility'].iloc[-period_days]) / data['30DayVolatility'].iloc[-period_days] * 100

volatility_changes = {
    "1 Month": calculate_volatility_change(data, 30),
    "3 Months": calculate_volatility_change(data, 90),
    "6 Months": calculate_volatility_change(data, 180),
    "1 Year": calculate_volatility_change(data, 365)
}

# Display current volatility and changes
st.markdown("### 30-Day Annualized Volatility")
st.markdown(f"**Current 30-Day Annualized Volatility**: {current_volatility:.2f}%")

volatility_change_cols = st.columns(len(volatility_changes))
for i, (label, change) in enumerate(volatility_changes.items()):
    if change is not None:
        color = "green" if change > 0 else "red"
        volatility_change_cols[i].markdown(f"<span style='color:{color}'>{label}: {change:.2f}%</span>", unsafe_allow_html=True)
    else:
        volatility_change_cols[i].markdown(f"{label}: N/A")

# Adjusted CVaR function for Long and simplified Short position
def calculate_cvar(returns, confidence_level=0.80, position_type="Long"):
    sorted_returns = np.sort(returns)
    
    if position_type == "Long":
        # For Long positions, focus on negative tail (lower returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        cvar = sorted_returns[:index].mean()  # Negative tail (lower bound)
    
    else:  # Short position
        # For Short positions, take the average of the top 20% of returns (above 80th percentile)
        index = int(confidence_level * len(sorted_returns))
        cvar = sorted_returns[index:].mean()  # Positive tail (upper bound)
    
    return cvar


# Add a radio button to select Long or Short position
position_type = st.sidebar.radio("Select Position Type", options=["Long", "Short"])

# Inputs for capital and risk percentage
capital = st.sidebar.number_input("Total Capital ($)", value=10000)
risk_percentage = st.sidebar.number_input("Percentage of Capital at Risk (%)", value=1.0)

# Fetch the last adjusted close price from data2
last_close_price = data2['Adj Close'].iloc[-1]

# Adjust the CVaR and Stop Loss calculations based on the position type
if position_type == "Long":
    # Calculate CVaR at 80% confidence level for Long positions (negative tail)
    cvar = calculate_cvar(monthly_returns, confidence_level=0.80, position_type="Long")

    # Position size calculation for Long
    risk_amount = (risk_percentage / 100) * capital
    position_size = risk_amount / abs(cvar) / last_close_price

    # Stop loss for Long position (stop loss below the price)
    stop_loss = last_close_price - (last_close_price * abs(cvar))

else:  # Short Position
    # For Short positions, calculate the average of the top 20% of returns (positive tail)
    cvar = calculate_cvar(monthly_returns, confidence_level=0.80, position_type="Short")

    # Position size calculation for Short
    risk_amount = (risk_percentage / 100) * capital
    position_size = risk_amount / abs(cvar) / last_close_price

    # Stop loss for Short position (stop loss above the price)
    stop_loss = last_close_price + (last_close_price * abs(cvar))

# Display the CVaR and position size
st.markdown("### Position Size Calculator")
st.markdown(f"**Position Type**: {position_type}")
st.markdown(f"**CVaR ({'80%' if position_type == 'Long' else '20%'} Confidence Level)**: {cvar:.2%}")
st.markdown(f"**Risk Amount**: ${risk_amount:.2f}")
st.markdown(f"**Position Size**: {position_size:.2f} units")
st.markdown(f"**Stop Loss**: {stop_loss:.2f}")
