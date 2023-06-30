#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:44:51 2023

@author: alpyildirim
"""

tickers = ['ADBE', 'ACN', 'AMD', 'ADI', 'ADSK', 'AKAM', 'ANET', 'ANSS', 'APH','CDW', 'CTSH', 'DXC','ENPH', 'EPAM',
           'FFIV', 'AAPL', 'AMAT', 'AVGO', 'FICO', 'FSLR', 'FTNT', 'GEN', 'GLW',
           'CDNS', 'CSCO', 'HPQ', 'INTC', 'IBM', 'LRCX', 'MSFT', 'HPE', 'INTU', 'IT', 'JNPR', 'KEYS',
           'NVDA', 'ORCL', 'QCOM', 'CRM', 'TXN', 'KLAC', 'MCHP', 'MPWR', 'MSI', 'MU', 'NOW',
           'NTAP', 'NXPI', 'ON', 'ORCL', 'PTC', 'QRVO', 'ROP', 'SEDG', 'SNPS', 'STX', 'SWKS', 'TDY', 'TEL', 'TER',
           'TRMB', 'TXN', 'TYL', 'VRSN', 'WDC', 'ZBRA', 'TSLA', 'GM', 'F']


import yfinance as yf
import pandas as pd

# Define the start and end dates
start_date = '2017-06-01'
end_date = '2023-06-11'

# Download the historical price data for all tickers
data = yf.download(tickers, start=start_date, end=end_date)

# Save the data to a CSV file
data.to_csv('historical_data.csv')

del(data)

# Read the CSV file into a DataFrame
df = pd.read_csv('historical_data.csv')
# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Set the date column as the index
df.set_index('Date', inplace=True)

import numpy as np

# Split the data into training and test sets
split_index = int(len(df) * 0.75)

train_set = df[:split_index]
test_set = df[split_index:]


# Initialize an empty matrix to store the cointegration test results
coint_matrix = np.zeros((len(train_set.columns), len(train_set.columns)))

from statsmodels.tsa.stattools import coint

# Calculate the cointegration test results for each pair of stocks
for i, stock_i in enumerate(train_set.columns):
    for j, stock_j in enumerate(train_set.columns):
        coint_test = coint(train_set[stock_i], train_set[stock_j])
        coint_matrix[i, j] = coint_test[1]  # Store the p-value in the matrix

# Convert the cointegration matrix to a DataFrame
coint_df_train = pd.DataFrame(coint_matrix, index=train_set.columns, columns=train_set.columns)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(
    coint_matrix, 
    xticklabels=train_set.columns, 
    yticklabels=train_set.columns, 
    cmap='RdYlGn_r', 
    mask = (coint_matrix >= 0.05)
)
plt.show()


#ANET and ON seems cointegrated with a very low p value, 0.016.
#ANET is a cloud networking company and ON has solutions to increase server powers. 
#Also considering that ANET is only cointegrated with ON out of all stocks, this cointegration seems valid.

#Let's check it further
import matplotlib.pyplot as plt

plt.plot(train_set['ANET'])
plt.plot(train_set['ON'])
plt.legend(['ANET', 'ON'])
plt.show()

#Calculating the slope and intercept with a Kalman Filter
delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)

from pykalman import KalmanFilter
import statsmodels.api as sm
obs_mat = sm.add_constant(np.log(train_set['ANET']).values, prepend=False)[:, np.newaxis]

kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)

state_means, state_covs = kf.filter(np.log(train_set['ON']).values)
slope=state_means[:, 0] 
intercept=state_means[:, 1]
plt.figure(figsize =(15,7))
plt.plot(train_set['ON'].index, slope, c='b')
plt.ylabel('slope')
plt.figure(figsize =(15,7))
plt.plot(train_set['ON'].index,intercept,c='r')
plt.ylabel('intercept')


# Calculate the Kalman Filter Spread
kalman_spread = np.log(train_set['ON']) - np.log(train_set['ANET']) * state_means[:,0] 

train_set_positions = pd.DataFrame()
train_set_positions['Spread'] = kalman_spread

#Check if the spread is stationary 
adf = sm.tsa.stattools.adfuller(train_set_positions['Spread'], maxlag=1)
print('ADF test statistic: %.02f' % adf[0])
for key, value in adf[4].items():
    print('\t%s: %.3f' % (key, value))
print('p-value: %.03f' % adf[1])

# We can reject the null hypothesis with a p-value very close to 0.
# Rejecting the null hypothesis means that the process has no unit root, and that the time series is stationary

# Plotting the spread
train_set_positions['Mean'] = train_set_positions['Spread'].mean()
train_set_positions['Upper'] = train_set_positions['Mean'] + train_set_positions['Spread'].std()
train_set_positions['Lower'] = train_set_positions['Mean'] - train_set_positions['Spread'].std()

train_set_positions.plot(figsize =(15,10),style=['g', '--r', '--b', '--b'])

# Add a new column 'Positions' and initialize it with zeros
train_set_positions['Positions'] = 0

# Initialize the position variable
position = 0

# Iterate through the rows of the DataFrame
for index, row in train_set_positions.iterrows():
    spread = row['Spread']
    mean_spread = row['Mean']
    lower_threshold = row['Lower']
    higher_threshold = row['Upper']

    # Check if the spread is below the lower threshold
    if spread < lower_threshold and position != 1:
        position = 1
    # Check if the spread is above the higher threshold
    elif spread > higher_threshold and position != -1:
        position = -1
    # Check if the spread is higher than the mean and the position is 1
    elif spread > mean_spread and position == 1:
        position = 0
    # Check if the spread is lower than the mean and the position is -1
    elif spread < mean_spread and position == -1:
        position = 0

    # Update the 'Positions' column
    train_set_positions.at[index, 'Positions'] = position

# Visualize the positions and spread together
fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(train_set_positions['Spread'], label='Spread')
ax.plot(train_set_positions['Mean'], label='Mean')
ax.plot(train_set_positions['Lower'], label='Lower')
ax.plot(train_set_positions['Upper'], label='Upper')
ax.plot(train_set_positions['Positions'], label='Positions')

# Add a legend and axis labels
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value')

# Show the plot
plt.show()


# Calculate ON log returns, add them to the dataframe
on_returns = np.log(train_set['ON'].pct_change() + 1)
on_returns.iloc[0] = 0
train_set_positions['ON Returns'] = on_returns

# Calculate ANET log returns, multiply them by beta(hedge ratio) and add them to the dataframe
anet_returns = np.log(train_set['ANET'].pct_change() + 1)
anet_returns.iloc[0] = 0
anet_returns_updated = anet_returns * state_means[:,0] 
train_set_positions['ANET Returns'] = anet_returns_updated
# Calculate daily strategy returns
train_set_positions['ON Strategy'] = train_set_positions['ON Returns'] * train_set_positions['Positions']
train_set_positions['ANET Strategy'] = train_set_positions['ANET Returns'] * -train_set_positions['Positions']
train_set_positions['Strategy Returns'] = train_set_positions['ON Strategy'] + train_set_positions['ANET Strategy']
# Calculate cumulative returns of the strategy
train_set_positions['Cumulative Strategy Returns'] = (1 + train_set_positions['Strategy Returns']).cumprod() - 1
# Calculate 'Gross PnL' based on the initial investment of 10000
train_set_positions['Gross PnL'] = 10000 * (1 + train_set_positions['Cumulative Strategy Returns'])

# Initialize 'Net PnL' column with the initial investment
train_set_positions['Net PnL'] = 10000
# Create a new column that indicates when the position changes
train_set_positions['Position Changes'] = train_set_positions['Positions'].diff().abs()

# Initialize a variable to hold the current balance
current_balance = 10000

# Iterate over each row in the DataFrame to add the net PnL
for i in train_set_positions.index:
    # Calculate the gross return for the current period
    gross_return = current_balance * train_set_positions.loc[i, 'Strategy Returns']
    
    # Check if the position has changed
    if train_set_positions.loc[i, 'Position Changes'] == 1:
        # Apply transaction costs
        transaction_cost = current_balance * 0.002
    else:
        transaction_cost = 0
    
    # Update the current balance
    current_balance = current_balance + gross_return - transaction_cost
    
    # Update the 'Net PnL' for the current period
    train_set_positions.loc[i, 'Net PnL'] = current_balance


# Visualize Gross and Net PnL evolution, with an initial balance of 10000
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(train_set_positions.index, train_set_positions['Gross PnL'], label='Cumulative Gross PnL')
ax.plot(train_set_positions.index, train_set_positions['Net PnL'], label='Cumulative Net PnL')

# Add axis labels and a title
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL')
ax.set_title('Cumulative PnL over Time')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Calculate the average return
average_return = train_set_positions['Strategy Returns'].mean()

# Calculate the standard deviation of the returns
return_std = train_set_positions['Strategy Returns'].std()

# Calculate the Sharpe Ratio
sharpe_ratio = average_return / return_std

# Annualize the Sharpe Ratio
trading_days = 252  # The number of trading days in a year
annualized_sharpe_ratio = sharpe_ratio * np.sqrt(trading_days)

############ Test set

obs_mat_test = sm.add_constant(np.log(test_set['ANET']).values, prepend=False)[:, np.newaxis]

kf_test = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat_test,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)

state_means_test, state_covs_test = kf_test.filter(np.log(test_set['ON']).values)
slope_test=state_means_test[:, 0] 
intercept_test=state_means_test[:, 1]
plt.figure(figsize =(15,7))
plt.plot(test_set['ON'].index, slope_test, c='b')
plt.ylabel('slope')
plt.figure(figsize =(15,7))
plt.plot(test_set['ON'].index,intercept_test,c='r')
plt.ylabel('intercept')

# The slope(alpha) and the intercept(beta) both increase throughout the test set


# Calculate the Kalman Filter Spread
kalman_spread_test = np.log(test_set['ON']) - np.log(test_set['ANET']) * state_means_test[:,0]


test_set_positions = pd.DataFrame()
test_set_positions['Spread'] = kalman_spread_test

# Check if the spread is stationary 
adf = sm.tsa.stattools.adfuller(test_set_positions['Spread'], maxlag=1)
print('ADF test statistic: %.02f' % adf[0])
for key, value in adf[4].items():
    print('\t%s: %.3f' % (key, value))
print('p-value: %.03f' % adf[1])

# We can reject the null hypothesis with a p-value very close to 0.
# Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.

# Plot the spread
test_set_positions['Mean'] = test_set_positions['Spread'].mean()
test_set_positions['Upper'] = test_set_positions['Mean'] + test_set_positions['Spread'].std()
test_set_positions['Lower'] = test_set_positions['Mean'] - test_set_positions['Spread'].std()

test_set_positions.plot(figsize =(15,10),style=['g', '--r', '--b', '--b'])

# Add a new column 'Positions' and initialize it with zeros
test_set_positions['Positions'] = 0

# Initialize the position variable
position = 0

# Iterate through the rows of the DataFrame
for index, row in test_set_positions.iterrows():
    spread = row['Spread']
    mean_spread = row['Mean']
    lower_threshold = row['Lower']
    higher_threshold = row['Upper']

    # Check if the spread is below the lower threshold
    if spread < lower_threshold and position != 1:
        position = 1
    # Check if the spread is above the higher threshold
    elif spread > higher_threshold and position != -1:
        position = -1
    # Check if the spread is higher than the mean and the position is 1
    elif spread > mean_spread and position == 1:
        position = 0
    # Check if the spread is lower than the mean and the position is -1
    elif spread < mean_spread and position == -1:
        position = 0

    # Update the 'Positions' column
    test_set_positions.at[index, 'Positions'] = position

# Plot the spread and the positions together
fig, ax = plt.subplots(figsize=(16, 12))

ax.plot(test_set_positions['Spread'], label='Spread')
ax.plot(test_set_positions['Mean'], label='Mean')
ax.plot(test_set_positions['Lower'], label='Lower')
ax.plot(test_set_positions['Upper'], label='Upper')
ax.plot(test_set_positions['Positions'], label='Positions')

# Add a legend and axis labels
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value')

# Show the plot
plt.show()


# Calculate ON log returns and add them to the dataframe
on_returns_test = np.log(test_set['ON'].pct_change() + 1)
on_returns_test.iloc[0] = 0
test_set_positions['ON Returns'] = on_returns_test

# Calculate ANET log returns, multiply by beta(hedge ratio) and add them to the dataframe
anet_returns_test = np.log(test_set['ANET'].pct_change() + 1)
anet_returns_test.iloc[0] = 0
anet_returns_updated_test = anet_returns_test * state_means_test[:,0] 
test_set_positions['ANET Returns'] = anet_returns_updated_test
# Calculate daily strategy returns
test_set_positions['ON Strategy'] = test_set_positions['ON Returns'] * test_set_positions['Positions']
test_set_positions['ANET Strategy'] = test_set_positions['ANET Returns'] * -test_set_positions['Positions']
test_set_positions['Strategy Returns'] = test_set_positions['ON Strategy'] + test_set_positions['ANET Strategy']
# Calculate cumulative returns of the strategy
test_set_positions['Cumulative Strategy Returns'] = (1 + test_set_positions['Strategy Returns']).cumprod() - 1
# Calculate 'Gross PnL' based on the initial investment of 10000
test_set_positions['Gross PnL'] = 10000 * (1 + test_set_positions['Cumulative Strategy Returns'])

# Initialize 'Net PnL' column with the initial investment
test_set_positions['Net PnL'] = 10000
# Create a new column that indicates when the position changes
test_set_positions['Position Changes'] = test_set_positions['Positions'].diff().abs()



# Initialize a variable to hold the current balance
current_balance = 10000

# Iterate over each row in the DataFrame to obtain the Net PnL
for i in test_set_positions.index:
    # Calculate the gross return for the current period
    gross_return = current_balance * test_set_positions.loc[i, 'Strategy Returns']
    
    # Check if the position has changed
    if test_set_positions.loc[i, 'Position Changes'] == 1:
        # Apply transaction costs
        transaction_cost = current_balance * 0.002
    else:
        transaction_cost = 0
    
    # Update the current balance
    current_balance = current_balance + gross_return - transaction_cost
    
    # Update the 'Net PnL' for the current period
    test_set_positions.loc[i, 'Net PnL'] = current_balance


# Plot the evolution of Gross and Net PnL for the test set
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(test_set_positions.index, test_set_positions['Gross PnL'], label='Cumulative Gross PnL')
ax.plot(test_set_positions.index, test_set_positions['Net PnL'], label='Cumulative Net PnL')

# Add axis labels and a title
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL')
ax.set_title('Cumulative PnL over Time')

# Add a legend
ax.legend()

# Show the plot
plt.show()

# Calculate the average return
average_return_test = test_set_positions['Strategy Returns'].mean()

# Calculate the standard deviation of the returns
return_std_test = test_set_positions['Strategy Returns'].std()

# Calculate the Sharpe Ratio
sharpe_ratio_test = average_return_test / return_std_test

# Annualize the Sharpe Ratio
annualized_sharpe_ratio_test = sharpe_ratio_test * np.sqrt(trading_days)
