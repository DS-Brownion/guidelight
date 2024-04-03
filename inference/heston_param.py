import numpy as np
from scipy.optimize import differential_evolution
import torch
import QuantLib as ql
# def heston_implied_volatility(parameters, market_vols, spot_prices):
#     """
#     Computes the implied volatilities from the Heston model.
#     """
# 	# Unpack the parameters
#     kappa, theta, sigma, rho, v0, r, q = parameters
#     heston_vols = []
#     for S, K, T in zip(spot_prices, strike_prices, maturity_dates):
#           heston_vols.append(heston_implied_vol(S, K, T, kappa, theta, sigma, rho, v0, r, q))

#     return np.array(heston_vols)

# def heston_objective_function(parameters, market_vols, spot_prices, strike_prices, maturity_dates):
#     """
#     Computes the loss function to be minimized.
#     """
#     heston_vols = heston_predictions(parameters, market_vols, spot_prices, strike_prices, maturity_dates)
#     loss = np.mean((market_vols - heston_vols) ** 2)
#     return loss

# def estimate_historical_parameters(market_vols, spot_prices, strike_prices, maturity_dates):
#     bounds = [(0, 10), (0.01, 0.5), (0.1, 1.5), (-1, 1), (0.001, 1), (0.01, 0.5)]
#     result = differential_evolution(heston_objective_function, bounds, args=(market_vols, spot_prices, strike_prices, maturity_dates))
#     estimated_parameters = result.x

#     return estimated_parameters

def estimate_daily_parameters(spot_prices, historical_vol, risk_free_rate_quote):
    bounds = [(0, 10), (0.01, 0.5), (0.1, 1.5), (-1, 1), (0.001, 1), (0.01, 0.5)]
    result = differential_evolution(heston_objective_function, bounds, args=(historical_vol, risk_free_rate_quote))

def heston_objective_function(params, historical_vol, risk_free_rate_quote):
    """
    Computes the loss function to be minimized.
    """
    simulated_vols = heston_model_volatility(params)
    loss = np.mean((simulated_vols - historical_vol) ** 2)
    return loss
def heston_model_volatility(parameters, risk_free_rate_quote):
    q = ql.SimpleQuote(risk_free_rate_quote)

    r = heston_predictions(parameters, q)

    return implied_volatility

def estimate_historical_volatility(spot_prices):
    """
    Estimates the historical volatility from the spot prices.
    """

    if not isinstance(spot_prices, torch.Tensor):
        spot_prices = torch.tensor(spot_prices, dtype=torch.float)
    elif spot_prices.dtype != torch.float:
        spot_prices = spot_prices.to(dtype=torch.float)

    # Calculate log returns
    log_returns = torch.log(spot_prices[1:] / spot_prices[:-1])
    
    # Calculate the standard deviation of log returns
    std_dev = torch.std(log_returns, unbiased=True)
    
    # Annualize the volatility, assuming 390 trading minutes in a day
    return std_dev * torch.sqrt(torch.tensor(390.0))



def heston_predictions(parameters, risk_free_rate_quote: ql.SimpleQuote, r, q, spot_price, length, time_steps):
    today = ql.Date().todaysDate()
    # daycount convention, not the difference in days
    daycount = ql.Actual360()
    calendar = ql.TARGET()


    kappa, theta, sigma, rho, v0  = parameters
    # Convert SimpleQuote to QuoteHandle for use in FlatForward
    risk_free_rate_handle = ql.QuoteHandle(r)

    # Using FlatForward with QuoteHandle and day count convention
    rf_r = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))
    d_y = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))

    heston_process = ql.HestonProcess(rf_r, d_y, ql.QuoteHandle(ql.SimpleQuote(spot_price)), v0, kappa, theta, sigma, rho)
    


    # Monte Carlo simulation of the Heston process
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(time_steps, ql.UniformRandomGenerator()))
    sequenceGen = ql.GaussianPathGenerator(heston_process, length, time_steps, rng, False)

    asset_paths = []
    vol_paths = []

    for i in range(time_steps):
        sample_path = sequenceGen.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        asset_path = [path[j] for j in range(len(path))]
        vol_path = [np.sqrt(v0) * np.exp(kappa * (theta - np.log(v0)) * t - 0.5 * sigma**2 * t) for t in time]  # Simplified
        
        asset_paths.append(asset_path)
        vol_paths.append(vol_path)

    return asset_path, vol_path
  