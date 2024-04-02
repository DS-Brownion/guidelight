import numpy as np
from scipy.optimize import differential_evolution
import torch
import QuantLib as ql
def heston_implied_volatility(parameters, market_vols, spot_prices, strike_prices, maturity_dates):
    """
    Computes the implied volatilities from the Heston model.
    """
	# Unpack the parameters
    kappa, theta, sigma, rho, v0, r, q = parameters
    heston_vols = []
    for S, K, T in zip(spot_prices, strike_prices, maturity_dates):
          heston_vols.append(heston_implied_vol(S, K, T, kappa, theta, sigma, rho, v0, r, q))

    return np.array(heston_vols)

def heston_objective_function(parameters, market_vols, spot_prices, strike_prices, maturity_dates):
    """
    Computes the loss function to be minimized.
    """
    heston_vols = heston_implied_volatility(parameters, market_vols, spot_prices, strike_prices, maturity_dates)
    loss = np.mean((market_vols - heston_vols) ** 2)
    return loss

def estimate_historical_parameters(market_vols, spot_prices, strike_prices, maturity_dates):
    bounds = [(0, 10), (0.01, 0.5), (0.1, 1.5), (-1, 1), (0.001, 1), (0.01, 0.5)]
    result = differential_evolution(heston_objective_function, bounds, args=(market_vols, spot_prices, strike_prices, maturity_dates))
    estimated_parameters = result.x

    return estimated_parameters

def heston_predictions(v0, kappa, theta, sigma, rho, risk_free_rate_quote: ql.SimpleQuote):
    today = ql.Date().todaysDate()
    # daycount convention, not the difference in days
    daycount = ql.Actual360()
    calendar = ql.TARGET()

    # Convert SimpleQuote to QuoteHandle for use in FlatForward
    risk_free_rate_handle = ql.QuoteHandle(risk_free_rate_quote)

    # Using FlatForward with QuoteHandle and day count convention
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))
    d_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, risk_free_rate_handle, daycount))

    hs_process = ql.HestonProcess(r_ts, d_ts, ql.QuoteHandle(ql.SimpleQuote(100)), v0, kappa, theta, sigma, rho)

    timestep = 50
    length = 0.5  # Simulation length in years
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(2*timestep, ql.UniformRandomGenerator()))
    times = list(ql.TimeGrid(length, timestep))
    seq = ql.GaussianMultiPathGenerator(hs_process, times, rng)

    sample_path = seq.next()
    multi_path = sample_path.value()
    print(type(multi_path[0]), type(multi_path[1]))
    # Extracting asset price and volatility paths
    asset_price_path = torch.tensor([multi_path[0][i] for i in range(len(times))])
    volatility_path = torch.tensor([multi_path[1][i] for i in range(len(times))])

    return asset_price_path, volatility_path
  