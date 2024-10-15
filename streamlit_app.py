import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Evaluating Returns on Stocks and shares',
    page_icon=':bar_chart:',  # This is an emoji shortcode for the bar chart emoji
    
# Define the path for your BigQuery credentials if needed
# from google.colab import auth
# auth.authenticate_user()

# Replace this with your actual BigQuery client setup if running locally
# from google.cloud import bigquery
# client = bigquery.Client()

# Load data from BigQuery or any other source
@st.cache_data
def load_data():
    # Replace this with your actual query
    # data = client.query('SELECT * FROM `risks-and-returrns-stocks.stocks_shares_csv.10yr_data`').to_dataframe()
    # For demonstration, simulate the DataFrame
    dates = pd.date_range(start="2014-10-17", end="2024-10-14", freq='D')
    data = pd.DataFrame({
        "DATE": dates,
        "AAPL": np.random.uniform(low=100, high=150, size=len(dates)),
        "NFLX": np.random.uniform(low=200, high=300, size=len(dates)),
        "BA": np.random.uniform(low=150, high=250, size=len(dates)),
        "T": np.random.uniform(low=25, high=35, size=len(dates)),
        "MGM": np.random.uniform(low=35, high=45, size=len(dates)),
        "AMZN": np.random.uniform(low=1500, high=3500, size=len(dates)),
        "IBM": np.random.uniform(low=100, high=140, size=len(dates)),
        "GOOG": np.random.uniform(low=1200, high=2800, size=len(dates)),
        "SP500": np.random.uniform(low=3000, high=4000, size=len(dates))
    })
    return data

data = load_data()

# Calculate daily returns for each stock and the market (S&P 500)
stocks = ['AAPL', 'NFLX', 'BA', 'T', 'MGM', 'AMZN', 'IBM', 'GOOG']

for stock in stocks:
    data[f'{stock}_Return'] = data[stock].pct_change()

data['SP500_Return'] = data['SP500'].pct_change()
data = data.dropna()

# Function to calculate beta using linear regression
def calculate_beta(stock_returns, market_returns):
    X = sm.add_constant(market_returns)  # Add a constant term (intercept)
    model = sm.OLS(stock_returns, X).fit()
    return model.params.iloc[1]  # Beta is the slope coefficient

# Calculate beta for each stock
beta_values = {}
market_returns = data['SP500_Return']

for stock in stocks:
    stock_returns = data[f'{stock}_Return']
    beta_values[stock] = calculate_beta(stock_returns, market_returns)

# Create a DataFrame to store the beta values
beta_df = pd.DataFrame(list(beta_values.items()), columns=['Stock', 'Beta'])
beta_df = beta_df.sort_values(by='Beta', ascending=False)

# Assumptions
risk_free_rate = 0.0409  # Assume 4.09% as the risk-free rate
market_avg_return = data['SP500_Return'].mean()

# Initialize an empty list to store stock beta, CAPM expected return, and the average stock return
stock_data = []

# Iterate through each stock to calculate its CAPM and average return
for stock in stocks:
    beta = beta_values[stock]
    capm_expected_return = risk_free_rate + beta * (market_avg_return - risk_free_rate)
    avg_stock_return = data[f'{stock}_Return'].mean()
    stock_data.append({
        'Stock': stock,
        'Beta': beta,
        'CAPM_Expected_Return': capm_expected_return,
        'Avg_Stock_Return': avg_stock_return
    })

# Convert the list of stock data into a DataFrame
stock_selection_df = pd.DataFrame(stock_data)
stock_selection_df = stock_selection_df.sort_values(by='CAPM_Expected_Return', ascending=False)

# Streamlit UI Components
st.title("Stock Analysis Dashboard")
st.write("This dashboard provides an analysis of various stocks and their performance based on CAPM.")

st.header("Stock Selection Data")
st.dataframe(stock_selection_df)

# Visualize Expected Returns and Average Returns
st.header("Expected Returns vs Average Returns")
fig = px.bar(stock_selection_df, x='Stock', y=['CAPM_Expected_Return', 'Avg_Stock_Return'], 
              title='Expected Returns vs Average Returns', barmode='group')
st.plotly_chart(fig)

# Function to normalize the prices based on the initial price
def normalize(df):
    normalized_df = df.copy()
    for col in normalized_df.columns[1:]:
        normalized_df[col] = normalized_df[col] / normalized_df[col].iloc[0]
    return normalized_df

# Normalize stock prices
normalized_data = normalize(data[["DATE"] + stocks])
st.header("Normalized Stock Prices")
fig = px.line(normalized_data, x='DATE', y=stocks, title='Normalized Stock Prices Over Time')
st.plotly_chart(fig)

# Create scatter plots with regression lines for each stock
for stock in stocks:
    st.header(f'Scatter Plot of {stock} Returns vs Market Returns')
    fig, ax = plt.subplots()
    plt.scatter(data['SP500_Return'], data[f'{stock}_Return'], alpha=0.5)
    
    # Fit regression line
    X = sm.add_constant(data['SP500_Return'])
    model = sm.OLS(data[f'{stock}_Return'], X).fit()
    plt.plot(data['SP500_Return'], model.predict(X), color='red', label=f'Fit Line (Beta={model.params[1]:.2f})')
    
    plt.title(f'Scatter Plot of {stock} Returns vs Market Returns')
    plt.xlabel('Market Returns (S&P 500)')
    plt.ylabel(f'{stock} Returns')
    plt.legend()
    plt.grid()
    
    # Use Streamlit to show the matplotlib figure
    st.pyplot(fig)

# Portfolio Beta Calculation
n = len(stocks)
weights = [1/n] * n
portfolio_beta = sum(weights[i] * beta_df['Beta'].iloc[i] for i in range(n))
st.header(f"Portfolio Beta: {portfolio_beta}")

# Additional Regression Analysis (Optional)
st.header("Regression Analysis Example")
X = data[['SP500_Return']]
y = data[[f'{stocks[0]}_Return']].copy()  # Use one stock return as an example

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate and display the mean squared error
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse:.4f}')

# Plot actual vs predicted values
fig, ax = plt.subplots()
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')  # Diagonal line
plt.title('Actual vs Predicted Returns')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.grid()
st.pyplot(fig)


