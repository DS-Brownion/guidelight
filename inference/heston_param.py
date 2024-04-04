import numpy as np
from scipy.optimize import differential_evolution
import torch
import QuantLib as ql

def calibrate_daily_parameters(historical_vol, risk_free_rate, spot_prices, q, time_steps, num_paths):
    bounds = [(0, 10), (0.01, 0.5), (0.1, 1.5), (-1, 1), (0.001, 1)]  # Bounds for kappa, theta, sigma, rho, v0
    result = differential_evolution(
        heston_objective_function,
        bounds,
        args=(historical_vol,risk_free_rate, spot_prices, q, time_steps, num_paths),  # Fixed parameters passed here
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=10
    )

    return result.x


def heston_objective_function(params, historical_vol, risk_free_rate, spot_prices, q, time_steps, num_paths):
    
    kappa, theta, sigma, rho, v0 = params
    median_simulated_vols = torch.zeros(spot_prices.shape[0], dtype=torch.float32)

    
    for i, price in enumerate(spot_prices):
        _, simulated_vols = heston_predictions(kappa, theta, sigma, rho, v0, risk_free_rate, q, price, 1/252, time_steps, num_paths)
        median_simulated_vols[i] = torch.median(torch.median(torch.tensor(simulated_vols), dim=0).values)  # Ensure conversion to tensor if needed

    return torch.mean((median_simulated_vols - torch.tensor(historical_vol)) ** 2)  # Ensure historical_vol is correctly handled




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



def heston_predictions(kappa, theta, sigma, rho, v0, risk_free_rate, q, spot_price, length, time_steps , num_paths = 1000):
    today = ql.Date().todaysDate()
    # daycount convention, not the difference in days
    daycount = ql.Actual360()



    # Convert SimpleQuote to QuoteHandle for use in FlatForward
    risk_free_rate_handle = ql.QuoteHandle(ql.SimpleQuote(risk_free_rate))
    q_handle = ql.QuoteHandle(ql.SimpleQuote(q))
    # Using FlatForward with QuoteHandle and day count convention
    rf_r = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))
    d_y = ql.YieldTermStructureHandle(ql.FlatForward(today, q_handle, daycount))

    heston_process = ql.HestonProcess(rf_r, d_y, ql.QuoteHandle(ql.SimpleQuote(spot_price)), v0, kappa, theta, sigma, rho)
    


    # Monte Carlo simulation of the Heston process
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(2*time_steps, ql.UniformRandomGenerator()))
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
  