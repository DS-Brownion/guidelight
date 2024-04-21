
import py_vollib_vectorized
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os, pickle
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from sklearn.preprocessing import MinMaxScaler
from multiprocess import Pool
from joblib import Parallel, delayed
import numpy as np
from numpy import sqrt, exp, pi, cos, sin, log, abs
from numba import njit

def heston_char(u, params):
    kappa, theta, zeta, rho, v0, r, q, T, S0 = params 
    t0 = 0.0 ;  q = 0.0
    m = np.log(S0) + (r - q)*(T-t0)
    D = np.sqrt((rho*zeta*1j*u - kappa)**2 + zeta**2*(1j*u + u**2))
    C = (kappa - rho*zeta*1j*u - D) / (kappa - rho*zeta*1j*u + D)
    beta = ((kappa - rho*zeta*1j*u - D)*(1-np.exp(-D*(T-t0)))) / (zeta**2*(1-C*np.exp(-D*(T-t0))))
    alpha = ((kappa*theta)/(zeta**2))*((kappa - rho*zeta*1j*u - D)*(T-t0) - 2*np.log((1-C*np.exp(-D*(T-t0))/(1-C))))
    return np.exp(1j*u*m + alpha + beta*v0)




@njit(parallel=True)
def Fourier_Heston_Put(S0, K, T, r, 
                  # Heston Model Paramters
                  kappa, # Speed of the mean reversion 
                  theta, # Long term mean
                  rho,   # correlation between 2 random variables
                  zeta,  # Volatility of volatility
                  v0,    # Initial volatility 
                  opt_type,
                  N = 1_012,
                  z = 24
                  ):

  def heston_char(u): 
    t0 = 0.0 ;  q = 0.0
    m = log(S0) + (r - q)*(T-t0)
    D = sqrt((rho*zeta*1j*u - kappa)**2 + zeta**2*(1j*u + u**2))
    C = (kappa - rho*zeta*1j*u - D) / (kappa - rho*zeta*1j*u + D)
    beta = ((kappa - rho*zeta*1j*u - D)*(1-exp(-D*(T-t0)))) / (zeta**2*(1-C*exp(-D*(T-t0))))
    alpha = ((kappa*theta)/(zeta**2))*((kappa - rho*zeta*1j*u - D)*(T-t0) - 2*log((1-C*exp(-D*(T-t0))/(1-C))))
    return exp(1j*u*m + alpha + beta*v0)
  
  # # Parameters for the Function to make sure the approximations are correct.
  c1 = log(S0) + r*T - .5*theta*T
  c2 = theta/(8*kappa**3)*(-zeta**2*exp(-2*kappa*T) + 4*zeta*exp(-kappa*T)*(zeta-2*kappa*rho) 
        + 2*kappa*T*(4*kappa**2 + zeta**2 - 4*kappa*zeta*rho) + zeta*(8*kappa*rho - 3*zeta))
  a = c1 - z*sqrt(abs(c2))
  b = c1 + z*sqrt(abs(c2))
  
  h       = lambda n : (n*pi) / (b-a) 
  g_n     = lambda n : (exp(a) - (K/h(n))*sin(h(n)*(a - log(K))) - K*cos(h(n)*(a - log(K)))) / (1 + h(n)**2)
  g0      = K*(log(K) - a - 1) + exp(a)
  
  F = g0 
  for n in range(1, N+1):
    h_n = h(n)
    F += 2*heston_char(h_n) * exp(-1j*a*h_n) * g_n(n)

  F = exp(-r*T)/(b-a) * np.real(F)
  F = F if opt_type == 'p' else F + S0 - K*exp(-r*T)
  return F if F > 0 else 0




S0      = 100.      # initial asset price
K       = 50.       # strike
r       = 0.03      # risk free rate
T       = 1/365     # time to maturity

v0=0.4173 ; kappa=0.4352 ; theta=0.2982 ; zeta=1.3856 ; rho=-0.0304



price = 0.10 ; S = 95 ; K = 100 ; t = .2 ; r = .2 ; flag = 'c'

def implied_volatility(price, S, K, t, r, flag):
  return py_vollib_vectorized.vectorized_implied_volatility(
    price, S, K, t, r, flag, q=0.0, on_error='ignore', model='black_scholes_merton',return_as='numpy') 


import pyfftw


import numpy as np
from numpy import exp, pi, log, sqrt
import pyfftw



def get_implied_volatility(price, S, K, t, r, flag):
    return py_vollib_vectorized.vectorized_implied_volatility(
        price, S, K, t, r, flag, q=0.0, on_error='ignore', model='black_scholes_merton',return_as='numpy') 


def print_debug_info(v0, kappa, theta, zeta, rho, wmae, idx, zeros, rmse):
    print(
        f">>v0={v0:.4f}; kappa={kappa:.4f}; theta={theta:.4f}; "
        f"zeta={zeta:.4f}; rho={rho:7.4f} | WMAE(IV): {wmae:.5e} | "
        f"Nulls: {idx.sum()}/{idx.shape[0]} | Zeros: {zeros}/{idx.shape[0]} | "
        f"WRMSE(IV): {rmse:.5e}"
    )
def SqErr(x, volSurface, _S, _K, _T, _r, _IV, _Weight):
    
    v0, kappa, theta, zeta, rho = x

    # Calculate prices using Heston Model
    Price_Heston = get_resutls_array_Heston(
        volSurface, v0, kappa, theta, zeta, rho, N=1_012, z=24
    )
    
    # Calculate implied volatilities
    IV_Heston = get_implied_volatility(
        price=Price_Heston, S=_S, K=_K, t=_T, r=_r, flag='p'
    )
    
    # Handle undefined IV calculations
    diff = IV_Heston - _IV
    idx = np.isnan(diff) | np.isinf(diff)
    diff[idx] = 0 - _IV[idx]
    IV_Heston[idx] = 0
    diff = np.nan_to_num(diff, 0)
    # Calculate RMSE
    rmse = sqrt(np.mean((diff * 100) ** 2 * _Weight))
    
    # Debugging info
    zeros = int(np.where(IV_Heston == 0, 1, 0).sum())
    wmae  = np.mean(np.abs(diff * 100) * _Weight)
    print_debug_info(v0, kappa, theta, zeta, rho, wmae, idx, zeros, rmse)
    return rmse
def get_error_Heston(volSurface, v0, kappa, theta, zeta, rho):
    """Calculates the error between the Heston model and the market prices.
    Arguments:
        volSurface: DataFrame with the market prices.
        v0: Initial variance.
        kappa: Mean reversion speed.
        theta: Long-run variance.
        zeta: Volatility of volatility.
        rho: Correlation between the variance and the asset.
    """
    error = 0
    for _, row in volSurface.iterrows():
        P = row['price']
        HP = Fourier_Heston_Put(S0=row['S'], K=row['strike'], v0=v0, kappa=kappa, theta=theta, zeta=zeta, rho=rho, T=row['maturity'], r=row['rate'], N=2048)
        error += (P - HP)**2

    return error / volSurface.shape[0]

def get_resutls_array_Heston(volSurface, v0, kappa, theta, zeta, rho, N=10_000, z=64):
    # Initialize the results array
    results = -np.ones(volSurface.shape[0])
    # reset the index of the options dataframe
    volSurface.index = np.arange(0, volSurface.shape[0])
    # loop through the rows of the options dataframe and run the Fourier_Heston_Put function
   
    for idx, row in volSurface.iterrows():
        results[idx] = Fourier_Heston_Put(S0=int(row['S']), K=int(row['strike']), v0=v0, kappa=kappa, theta=theta, zeta=zeta, rho=rho, T=row['maturity'], r=row['rate'], N=N, opt_type='p',z=z)
    return results

def get_resutls_df_Heston(volSurface, v0, kappa, theta, zeta, rho, N=2048, z=100):
    observed = volSurface.copy(deep=True)
    heston = volSurface.copy(deep=True)
    observed['source'] = 'Observed'
    heston['source'] = 'Heston Model'

    heston_prices = [] 
    implied_volatilities = []
    for _, row in volSurface.iterrows():
        heston_price = Fourier_Heston_Put(S0=row['S'], K=row['strike'], v0=v0, kappa=kappa, theta=theta, zeta=zeta, rho=rho, T=row['maturity'], r=row['rate'], N=N, opt_type='p', z=z)
        heston_prices.append(heston_price)
        # np.array(... , ndmin=1) So the type of the input is compatible with what numba expects
        maturity  = np.array(row['maturity'],ndmin=1)
        observed_price  = np.array(heston_price,ndmin=1)
        S0 = np.array(row['S'],ndmin=1)
        K  = np.array(row['strike'],ndmin=1)
        r  = np.array(row['rate'],ndmin=1)
        implied_volatility = get_implied_volatility(price=observed_price, S=S0, K=K, t=maturity, r=r, flag=option_type)
        implied_volatilities.append(implied_volatility[0])

    heston['price'] = heston_prices
    heston['IV']    = implied_volatilities

    return pd.concat([observed, heston])

def get_error_df_Heston(volSurface, v0, kappa, theta, zeta, rho, diff='Price', error='Error', weighted=True, N=10_000, z=64):
    if   error == 'Error':          _name = f'Weighted Error {diff}'             if weighted else f'Error {diff}'
    elif error == 'Perc Error':     _name = f'Weighted Persentage Error {diff}'  if weighted else f'Persentage Error {diff}'
    elif error == 'Squared Error':  _name = f'Weighted Squared Error {diff}'     if weighted else f'Squared Error {diff}'
    else: raise Exception("Error: variable 'error' is not defined correctly")
    
    results_df = {'strike':[], 'maturity':[], _name:[], 'Opt. Type':[], 'Weight':[]}

    for _, row in volSurface.copy(deep=True).iterrows():
        _P = Fourier_Heston_Put(S0=row['S'], K=row['strike'], v0=v0, kappa=kappa, theta=theta, zeta=zeta, rho=rho, T=row['maturity'], r=row['rate'], N=N, z=z, opt_type=row['Type'])
        # np.array(... , ndmin=1) So the type of the input is compatible with what numba expects
        _T  = np.array(row['maturity'],ndmin=1)
        _C  = np.array(_P,ndmin=1)
        _P  = np.array(row['price'],ndmin=1)
        _S0 = np.array(row['S'],ndmin=1)
        _K  = np.array(row['strike'],ndmin=1)
        _r  = np.array(row['rate'],ndmin=1)

        _IV  = get_implied_volatility(price=_C, S=_S0, K=_K, t=_T, r=_r, flag='p')
        _IV2 = get_implied_volatility(price=row['price'], S=_S0, K=_K, t=_T, r=_r, flag='p')

        if error    == 'Error':
            if diff == 'IV':  _error  = (_IV - _IV2) *                (row['Weight'] if weighted else 1)
            else           :  _error  = (_C - _P) *                   (row['Weight'] if weighted else 1)
        elif error  == 'Perc Error':
            if diff == 'IV':  _error  = ((_IV - _IV2)/_IV2) * 100 *   (row['Weight'] if weighted else 1)
            else           :  _error  = ((_C - _P)/_P) * 100 *        (row['Weight'] if weighted else 1)
        elif error  == 'Squared Error':
            if diff == 'IV':  _error  = (_IV - _IV2)**2 *             (row['Weight'] if weighted else 1)
            else           :  _error  = (_C - _P)**2 *                (row['Weight'] if weighted else 1)

        results_df[_name].append(_error[0])
        results_df['maturity'].append(_T[0])
        results_df['strike'].append(_K[0])
        results_df['Weight'].append(row['Weight']*10)

    return pd.DataFrame(results_df)


def heston_volSurface(cleaned_df, yields):
 
    volSurface = cleaned_df.drop(columns=['days since last trade', 'ticker'])
   

    

    def implied_volatility(price, S, K, t, r, flag):

        return py_vollib_vectorized.vectorized_implied_volatility(
            price, S, K, t, r, flag, q=0.0, on_error='ignore', model='black_scholes_merton',return_as='numpy')

    yield_maturities = np.array([1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    # yields  = np.array([5.30,5.39,5.50,5.50,5.44,5.11,4.33,3.98,3.70,3.66,3.61,3.98,3.84])
    # get the first row of the yield rates
    yield_rates = pd.read_csv("five-year-rates.csv")
    d = datetime.strftime(datetime.strptime("2023-04-18", "%Y-%m-%d"), "%m/%d/%Y")
    yields = yield_rates.loc[yield_rates["Date"]==d].values[:,1:].astype(np.float64).reshape(-1)
    scaler = MinMaxScaler()
    yields_normalized = scaler.fit_transform(yields.reshape(-1, 1)).flatten()
    curve_fit, _ = calibrate_nss_ols(yield_maturities, yields_normalized)
    volSurface['rate'] = volSurface['maturity'].apply(curve_fit) / 100
    volSurface['IV'] = implied_volatility(volSurface['price'], volSurface['S'], volSurface['strike'], volSurface['maturity'], volSurface['rate'], 'p')
    return volSurface

def heston_daily_volSurface(underlying_ticker, date):
    cleaned = daily_option_data(underlying_ticker, date)
    yield_rates = pd.read_csv("five-year-rates.csv")
   
    d = datetime.strftime(datetime.strptime(date, "%Y-%m-%d"), "%m/%d/%Y")
    yields = yield_rates.loc[yield_rates["Date"]==d].values[:,1:].astype(np.float64).reshape(-1)
    
    volSurface = heston_volSurface(cleaned, yields)
    return volSurface


def heston_parameters(VolSurface):
	# Extract data from dailyVolSurface DataFrame
    _K = VolSurface['strike'].to_numpy()

    _C = VolSurface['price'].to_numpy()
    _T      = VolSurface['maturity'].to_numpy()
    _r      = VolSurface['rate'].to_numpy()
    _S      = VolSurface['S'].to_numpy()
    _IV     = VolSurface['IV'].to_numpy()
    _Weight = VolSurface['Weight'].to_numpy()
    # Initial parameters and bounds for optimization
    params = {
        "v0": {"x0": np.random.uniform(1e-3, 1.2), "lbub": [1e-3, 1.2]},
        "kappa": {"x0": np.random.uniform(1e-3, 10), "lbub": [1e-3, 10]},
        "theta": {"x0": np.random.uniform(1e-3, 1), "lbub": [1e-3, 1.2]},
        "zeta": {"x0": np.random.uniform(1e-2, 4), "lbub": [1e-2, 4]},
        "rho": {"x0": np.random.uniform(-1, 1), "lbub": [-1, 1]}
    }
    x0 = [param["x0"] for _, param in params.items()]
    bnds = [param["lbub"] for _, param in params.items()]
    result = minimize(
    SqErr, x0, args=(VolSurface,  _S, _K, _T, _r, _IV, _Weight),  tol=1e-5, method='SLSQP',
    options={'maxiter': 80, 'ftol': 1e-5, 'disp': True},
    bounds=bnds, jac='3-point'
	)

    return result.x











