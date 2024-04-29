# Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pypfopt import EfficientFrontier, objective_functions
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import DiscreteAllocation

from pricing_model import CNNLSTMModel, huber_loss_with_fft
import torch

# Function to get stock data
def get_stock_data(symbols, period='20y'):
    return yf.download(symbols, period=period)['Adj Close']

# Function to get market capitalization
def get_market_capitalizations(symbols):
    mcaps = {}
    for t in symbols:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]
    return mcaps

# Function to calculate covariance matrix
def calculate_covariance_matrix(portfolio):
    returns = portfolio.pct_change().dropna()
    cov_matrix = returns.cov()
    return cov_matrix

# Function to calculate risk aversion
def calculate_risk_aversion(market_prices):
    return black_litterman.market_implied_risk_aversion(market_prices)

# Function to create Black-Litterman model
def create_black_litterman_model(covariance_matrix, market_prior, viewdict, omega=None):
    return BlackLittermanModel(covariance_matrix, pi=market_prior, absolute_views=viewdict, omega=omega)

# Function to create a heatmap
def create_heatmap(cov_matrix):
    sns.heatmap(cov_matrix.corr(), cmap='coolwarm')
    plt.show()

# Function to plot returns
def plot_returns(rets_df):
    rets_df.plot.bar(figsize=(12, 8))
    plt.show()

# Function to create a portfolio allocation
def create_portfolio_allocation(ret_bl, S_bl):
    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights = ef.clean_weights()
    return weights

# Main function
def main():
    # Define the stock symbols
    symbols = ['PFE', 'GSK', 'AZN', 'BMY', 'MRK', 'RHHBY', 'SNY', 'NVS', 'ABBV', 'JNJ']

    # Get the stock data and SP500 ETF benchmark
    portfolio = get_stock_data(symbols)
    market_prices = get_stock_data("SPY")

    # Get market capitalizations
    mcaps = get_market_capitalizations(symbols)

    # Calculate covariance matrix and risk aversion
    S = calculate_covariance_matrix(portfolio)
    delta = calculate_risk_aversion(market_prices)

    # Define prior returns and create Black-Litterman model
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    viewdict = {'PFE': 0.1, 'GSK': 0.1, 'AZN': -0.05, 'BMY': 0.25, 'MRK': 0.09, 'RHHBY': -0.12, 'SNY': 0.07,
                'NVS': -0.21, 'ABBV': 0.21, 'JNJ': 0.2}

    # Compute omega matrix based on confidence intervals
    intervals = [
        (0, 0.25),
        (0.1, 0.4),
        (-0.1, 0.15),
        (-0.05, 0.1),
        (0.03, 0.1),
        (-0.1, 0),
        (0.1, 0.2),
        (-0.08, 0.12),
        (0.1, 0.9),
        (0, 0.3)
    ]
    variances = [(ub - lb) / 2 ** 2 for lb, ub in intervals]
    omega = np.diag(variances)

    # Create Black-Litterman model and posterior returns
    bl = create_black_litterman_model(S, market_prior, viewdict, omega)
    ret_bl = bl.bl_returns()

    # Display posterior returns and create heatmap
    rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)],
                          index=["Prior", "Posterior", "Views"]).T
    plot_returns(rets_df)

    # Create portfolio allocation
    weights = create_portfolio_allocation(ret_bl, S)
    pd.Series(weights).plot.pie(figsize=(7, 7))
    plt.show()

    # Setup model with Torch
    model = CNNLSTMModel(num_features=10, num_output_features=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    for epoch in range(num_epochs):
        for batch in train_loader:
            x, y = batch  # Assuming x and y are prepared correctly
            optimizer.zero_grad()
            y_pred = model(x)
            loss = huber_loss_with_fft(y, y_pred)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
if __name__ == '__main__':
    main()