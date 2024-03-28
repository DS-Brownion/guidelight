
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import QuantLib as ql
from scipy.optimize import differential_evolution
from heston_param import estimate_historical_parameters, heston_predictions, heston_predictions





def daily_heston_parameters(daily_market_vols, daily_strike_prices, daily_spot_prices, daily_maturity_dates):
	daily_heston_parameter = []
	
		# Estimate Heston model parameters for each day
	parameters = estimate_historical_parameters(daily_market_vols, daily_strike_prices, daily_spot_prices, daily_maturity_dates)
	daily_heston_parameters.append(parameters)
	return daily_heston_parameter
def calculate_hist_parameters(market_vols, spot_prices, strike_prices, maturity_dates, num_days):
	for i in range(num_days):
		daily_parameter = daily_heston_parameters(market_vols[i], spot_prices[i], strike_prices[i], maturity_dates[i])
		daily_heston_parameters.append(daily_parameter)

	return daily_heston_parameters

class heston(nn.Module):
	def __init__(self):
		super(heston, self).__init__()
		self.mu = 0.0
		self.kappa = 0.0
		self.theta = 0.0
		self.xi = 0.0
		self.rho = 0.0

	def forward(self, params):
		self.mu, self.kappa, self.theta, self.xi, self.rho = params
		sp_path, v_path = heston_predictions(self.mu, self.kappa, self.theta, self.xi, self.rho)
		s_mean = sp_path.mean()
		p_mean = sp_path.mean()
		s_dev = sp_path - s_mean
		v_dev = v_path - p_mean

		covariance = (s_dev * v_dev).mean()
		e_x = s_mean
		beta = covariance / s_dev.std()
		return e_x, beta
	

class param_LSTM(nn.Module):
	def __init__(self, device='cpu'):
		super(param_LSTM, self).__init__()
		# utilizing 3 layers of LSTM
		self.lstms = nn.ModuleList([
            nn.LSTM(128, hidden_size=64, batch_first=True),
            nn.LSTM(64, hidden_size=32, batch_first=True),
            nn.LSTM(32, hidden_size=16, batch_first=True)  # Ensure consistency in input_size
        ])
		self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(2)])

		self.device = device
		self.linear = nn.Linear(16, 1)
		self.to(device)
	
	def forward(self,params):
		
		# since we would only need the last output of the LSTM
		
		params = params.to(self.device)
		for i in range(len(self.lstms)):
			params, _ = self.lstms[i](params)
			# apply residual connection
			if i < len(self.dropouts):
				params = self.dropouts[i](params)
		params = params[ -1, :]
		params = self.linear(params)
		
		return params
	
	def train_loop(self, train_h, train_eb, val_h, val_eb):
		# chooses gpu if available, otherwise cpu
		
		# chose the adam optimizer for the gradient descent
		train_h, train_eb = train_h.to(self.device), train_eb.to(self.device)
		val_h, val_eb = val_h.to(self.device), val_eb.to(self.device)
		optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
		# currently using MSE loss function to fit the model ()
		loss_fn = nn.MSELoss()

		# training the model for 100 epochs
		for epoch in range(100):
			# set to training mode
			self.train()

			# zero the gradients otherwise they would accumulate
			optimizer.zero_grad()
			pred = self(train_h)
			loss = loss_fn(pred, train_eb)
			loss.backward()
			optimizer.step()


			# set to evaluation mode
			self.eval()
			with torch.no_grad():  # Inference mode, gradients not computed
				val_pred = self(val_h)
				val_loss = loss_fn(val_pred, val_eb)
			
			print(f'Epoch {epoch} loss: {loss.item()}', f'Validation loss: {val_loss.item()}')


class neuralized_heston(nn.Module):
	def __init__(self, device='cpu'):
		super(neuralized_heston, self).__init__()
		self.heston = heston()
		self.param_LSTM = param_LSTM(device)
		self.device = device
		self.to(device)
	
	def forward(self, params):
		params = self.param_LSTM(params)
		return self.heston(params)


