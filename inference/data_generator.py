import py_vollib_vectorized
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from polygon import OptionsClient
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import os, pickle
from scipy.optimize import minimize
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from sklearn.preprocessing import MinMaxScaler
from multiprocess import Pool
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from heston_fft import heston_daily_volSurface, heston_parameters
import pickle


datetime_diff = lambda date1, date2 : (datetime.strptime(date1, '%Y-%m-%d') - datetime.strptime(date2, '%Y-%m-%d')).days


load_dotenv("/Users/brad/mlprojects/guidelight/guidelight-api/.env")
polygon_token = os.getenv("POLYGON_TOKEN")





import pandas as pd

if not os.path.exists("data.pkl"):
    indices = [(contract.ticker, contract.expiration_date, contract.strike_price) for contract in all_contracts]
    data = {}
    for index in indices:
        ticker, expiration_date, strike_price = index
        current_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        past_date = current_date - timedelta(days=14)

        # get key value data for each agg

        a = [vars(agg) for agg in client.get_aggs(ticker, 1, 'day', past_date, current_date)]
        data[index] = a


    pickle.dump(data, open("data.pkl", "wb"))

else:
    data = pickle.load(open("data.pkl", "rb"))


def get_agg_worker(agg):
    return vars(agg)

def generate_option_aggs(ticker, token=polygon_token):
    client = OptionsClient(api_key=token)
    if not os.path.exists(f"cache/{ticker}_contracts.pkl"):
        reqs = client.list_options_contracts(ticker,as_of="2024-04-16", expired=True, expiration_date_gt="2023-04-16")
        all_contracts = list(reqs)
        pickle.dump(all_contracts, open(f"cache/{ticker}_contracts.pkl", "wb"))
    else:
        all_contracts = pickle.load(open(f"cache/{ticker}_contracts.pkl", "rb"))



    indices = [(contract.ticker, contract.expiration_date, contract.strike_price) for contract in all_contracts]
    data = {}
    if not os.path.exists("data.pkl"):
        for index in indices:
            ticker, expiration_date, strike_price = index
            current_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            past_date = current_date - timedelta(days=14)

            # Fetch aggregates for each contract within the date range
            aggs = client.get_aggs(ticker, 1, 'day', past_date, current_date)
            
            # Using Pool for asynchronous map
            with Pool(10) as p:
                async_result = p.map_async(get_agg_worker, aggs)
                p.close()  # No more tasks will be submitted, safe to close the pool
                p.join()  # Wait for all worker processes to finish
                
                # Collect results
                results = async_result.get()
                data[index] = results
    else:
        data = pickle.load(open("data.pkl", "rb"))
    return data


def save_option_ticker(underlying_ticker:str, data):
# Flatten the data while preserving the option ticker and expiration date
	flattened_data = []
	for (ticker, expiration, strike_price), entries in data.items():
		for entry in entries:
			entry.update({
				"ticker": ticker,
				"expiration_date": expiration,
				"strike_price": strike_price
			})
			flattened_data.append(entry)

	# Create a DataFrame
	df = pd.DataFrame(flattened_data)

	# Set a MultiIndex using the ticker, expiration date, and trading date
	df.set_index(['ticker', "strike_price", 'expiration_date'], inplace=True)

	# get by ticker
	# 1681099200000
	df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime("%Y-%m-%d")
	# df.index = df.index.set_levels(pd.to_datetime(df.index.get_level_values('timestamp'), unit='ms').strftime('%Y-%m-%d %H:%M:%S'), level='timestamp')
	df.to_csv(f"options_contracts/{underlying_ticker.upper()}.csv", index_label=['ticker', "strike_price", 'expiration_date'])


# Flatten the data while preserving the option ticker and expiration date
flattened_data = []
for (ticker, expiration, strike_price), entries in data.items():
    for entry in entries:
        entry.update({
            "ticker": ticker,
            "expiration_date": expiration,
            "strike_price": strike_price
        })
        flattened_data.append(entry)

# Create a DataFrame
df = pd.DataFrame(flattened_data)

# Set a MultiIndex using the ticker, expiration date, and trading date
df.set_index(['ticker', "strike_price", 'expiration_date'], inplace=True)

# get by ticker
# 1681099200000
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime("%Y-%m-%d")
# df.index = df.index.set_levels(pd.to_datetime(df.index.get_level_values('timestamp'), unit='ms').strftime('%Y-%m-%d %H:%M:%S'), level='timestamp')


def daily_option_data(underlying_ticker:str, date:str):
	if os.path.exists(f"options_contracts/{underlying_ticker}-{date}.csv"):
		return pd.read_csv(f"options_contracts/{underlying_ticker}-{date}.csv")

	df = pd.read_csv(f'options_contracts/{underlying_ticker}.csv', index_col=[0, 1, 2])
	option_contracts = df.loc[df['timestamp'] == date]
	option_contracts.reset_index(inplace=True)
	# print(option_contracts)
	colnames = ["ticker", "maturity", "Weight", 'price', 'days since last trade', 'strike', 'S']
	volsurface = pd.DataFrame(columns=colnames)

	for ticker in option_contracts['ticker'].unique():
		agg_series= df.loc[(ticker, slice(None), slice(None))]
		i = np.where(agg_series['timestamp'].values == date)[0][0]
	# agg_series
		if i <= 0:
			continue

		diff = datetime_diff(agg_series['timestamp'].iloc[i], agg_series['timestamp'].iloc[i-1])
		if diff <= 3:
			expiration_date = agg_series.index.get_level_values(1).unique()[0]
			time_to_maturity =datetime_diff(expiration_date, date) 
			row = pd.DataFrame({
				'ticker': ticker,
				'maturity': time_to_maturity/365 if time_to_maturity else 6.5/(24 * 365),
				'price': agg_series["vwap"].values[i],
				'Weight': agg_series["volume"].values[i] / agg_series["volume"].sum(),
				'days since last trade': diff,
				'strike': agg_series.index.get_level_values(0).unique()[0],
				'S': agg_series['open'].values[i]
			}, columns=colnames, index=[0])

			volsurface = pd.concat([volsurface, row], ignore_index=True)
			

	
	volsurface.to_csv( os.path.join(os.getcwd(), f"options_data/{underlying_ticker}-{date}.csv"))
	return volsurface

# # Multiprocessing 



def process_multiple_days(underlying_ticker, start_date, end_date):
    # Generate list of dates
    dates = mcal.get_calendar("NYSE").valid_days(start_date=start_date, end_date=end_date)
    
    # Define a helper to wrap your existing function for use with starmap
    

    # Setup multiprocessing pool
    # with Pool() as pool:
    #     pool.starmap(worker, [(underlying_ticker, date) for date in dates])

    # print("Data processing complete for all specified dates.")
    dataset = [daily_option_data(underlying_ticker, timestamp.date().strftime("%Y-%m-%d")) for timestamp in dates]

    # return dataset
    return dataset

def heston_day_params(underlying_ticker, date):
    volSurface = heston_daily_volSurface(underlying_ticker, date)
    return heston_parameters(volSurface)


def heston_params(underlying_ticker, start_date, end_date):
	dates = mcal.get_calendar("NYSE").valid_days(start_date=start_date, end_date=end_date)
	# params = [heston_day_params(underlying_ticker, date.date().strftime("%Y-%m-%d")) for date in dates]
	df = pd.DataFrame(columns=["date", 'v0', 'kappa', 'theta', 'zeta', 'rho'])
	for date in dates:
		print("optimizing for", date.date().strftime("%Y-%m-%d"), f"for {underlying_ticker}")
		tries = 0
		while tries < 3:
			try:

				params = heston_day_params(underlying_ticker, date.date().strftime("%Y-%m-%d"))
				row = pd.DataFrame({
					"date": date.date().strftime("%Y-%m-%d"),
					'v0': params[0],
					'kappa': params[1],
					'theta': params[2],
					'zeta': params[3],
					'rho': params[4]
				}, columns=["date", 'v0', 'kappa', 'theta', 'zeta', 'rho'], index=[0])
				df = pd.concat([df, row])
				break
			except LinAlgError:
				print("LinAlgError, trying again")
				tries += 1
				continue

	return df



def ticker_to_x(dates, ticker, market_index_data, treasury_rates, earnings_per_share, annual_dividend):
    # Fetch data using your client method
    data = client.get_aggregate_bars(ticker, '2023-04-11', '2024-04-12', multiplier=1, timespan='day')["results"]
    data_df = pd.DataFrame(data)
    # get row by dates
    # Assuming you have a DataFrame called 'data_df' with a 'date' column
    # and you want to get the row for a specific date '2023-04-11'


    data_df["t"] = pd.to_datetime(data_df['t'], unit='ms').dt.strftime("%Y-%m-%d")
    # print(data_df)
    # # Join market index data and treasury rates on timestamps
    # data_df['market_returns'] = market_index_data['market_returns']
    # data_df['risk_free_rate'] = treasury_rates['6 Mo']
    # Assuming you have a DataFrame called 'data_df' with a 'date' column
    # dates = ['2023-04-11', '2023-04-12', '2023-04-13']

    # Filter rows based on the dates
    data_df = data_df.loc[data_df['t'].isin(dates)]

    print(data_df)
    # Join market index data and treasury rates on timestamps
    data_df['market_returns'] = market_index_data['market_returns']
    data_df['risk_free_rate'] = treasury_rates['6 Mo']
    
    # Prepare a list to hold tensor data
    tensor_data = []
    returns = data_df['c'].pct_change().dropna()
    # Loop over dataframe to compute features per day
    # print(returns)
    for index, row in data_df.iterrows():
        

        market_return = row['market_returns']
        

        # Beta calculation
        covariance = returns.cov(market_index_data)
        market_variance = market_index_data['market_returns'].var()
        beta = covariance / market_variance if market_variance != 0 else 0

        # Alpha calculation
        risk_free_rate = row['risk_free_rate']
        alpha = returns.mean() - (risk_free_rate + beta * (market_index_data['market_returns'].mean() - risk_free_rate))

        # Sharpe Ratio calculation
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Volatility calculation
        volatility = returns.std()

        # P/E Ratio calculation
        pe_ratio = row['c'] / earnings_per_share if earnings_per_share != 0 else 0

        # Dividend Yield calculation
        div_yield = (annual_dividend / row['c']) * 100 if row['c'] != 0 else 0

        # Append the calculated values to the list
        tensor_data.append([beta, alpha, sharpe_ratio, volatility, pe_ratio, div_yield])

    # Convert list to PyTorch tensor
    tensor_xvec = torch.tensor(tensor_data, dtype=torch.float32)

    return tensor_xvec



# the function to import the parameters for a company from param_data by importing from corresponding "ticker".csv file as time series of torch tensors
def import_params(ticker):
	# read the csv file
    df = pd.read_csv('param_data/' + ticker + '.csv')
    dates = df['date']
    columns = df.columns[2:]
	# turn rows into tensors and append them to the the tensor
    params = [tensor(df[col].values) for col in columns]
    params = torch.stack(params, dim=1)
    
    return params, dates