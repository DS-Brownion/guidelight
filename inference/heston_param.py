import numpy as np
from scipy.optimize import minimize
import torch
import QuantLib as ql

def calibrate_daily_parameters(historical_vol, risk_free_rate, stock_prices, q, time_steps, num_paths):
    bounds = [(0, 10), (0.01, 0.5), (0.1, 1.5), (-1, 1), (0.001, 1)]
    x0c = [np.random.uniform(low, high) for low, high in bounds]  # Bounds for kappa, theta, sigma, rho, v0
    
    result = minimize(heston_objective_function, x0=x0c, args=(historical_vol, risk_free_rate, stock_prices, q, time_steps, num_paths), bounds=bounds, method='L-BFGS-B')
    return result.x


def heston_objective_function(params, historical_vol, risk_free_rate, stock_prices, q, time_steps, num_paths):
    
    kappa, theta, sigma, rho, v0 = params
    

    simulated_prices, simulated_vols = heston_predictions(kappa, theta, sigma, rho, v0, risk_free_rate, q, stock_prices[0], 1/252, time_steps, num_paths)
    
    mean_simulated_prices = simulated_prices.mean(dim=0)[:-1]
    mean_simulated_vols = simulated_vols.mean(dim=0)[:-1]
    

    # Compute the MSE between the mean simulated paths and the historical data
    price_error = torch.mean((mean_simulated_prices - stock_prices) ** 2)
    vol_error = torch.mean((mean_simulated_vols - historical_vol) ** 2)  # Assumes historical_vol is already in a comparable form

    # Sum of the errors
    total_error = np.sqrt((price_error.item()) ** 2 + (vol_error.item()) ** 2)

    return total_error

def estimate_historical_volatility(stock_prices):
    """
    Estimates the historical volatility from the spot prices.
    """

    if not isinstance(stock_prices, torch.Tensor):
        stock_prices = torch.tensor(stock_prices, dtype=torch.float)
    elif stock_prices.dtype != torch.float:
        stock_prices = stock_prices.to(dtype=torch.float)

    # Calculate log returns
    log_returns = torch.log(stock_prices[1:] / stock_prices[:-1])
    
    # Calculate the standard deviation of log returns
    std_dev = torch.std(log_returns, unbiased=True)
    
    # Annualize the volatility, assuming 390 trading minutes in a day
    return std_dev * torch.sqrt(torch.tensor(390.0))



def heston_predictions(kappa, theta, sigma, rho, v0, risk_free_rate, q, stock_price, length, time_steps , num_paths = 1000):
    today = ql.Date().todaysDate()
    # daycount convention, not the difference in days
    daycount = ql.Actual360()



    # Convert SimpleQuote to QuoteHandle for use in FlatForward
    risk_free_rate_handle = ql.QuoteHandle(ql.SimpleQuote(risk_free_rate))
    q_handle = ql.QuoteHandle(ql.SimpleQuote(q))
    # Using FlatForward with QuoteHandle and day count convention
    rf_r = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))
    d_y = ql.YieldTermStructureHandle(ql.FlatForward(today, q_handle, daycount))

    heston_process = ql.HestonProcess(rf_r, d_y, ql.QuoteHandle(ql.SimpleQuote(stock_price)), v0, kappa, theta, sigma, rho)
    


    # Monte Carlo simulation of the Heston process
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(2 * time_steps, ql.UniformRandomGenerator()))
    times = list(ql.TimeGrid(length, time_steps))
    sequenceGen = ql.GaussianMultiPathGenerator(heston_process, times, rng)

    asset_paths = []
    vol_paths = []

    for i in range(num_paths):
        sample_path = sequenceGen.next()
        path = sample_path.value()
        
        asset_paths.append(path[0])
        vol_paths.append(path[1])

    return torch.FloatTensor(asset_paths), torch.FloatTensor(vol_paths)
  