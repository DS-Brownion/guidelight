import yfinance as yf

ticker = yf.download('MRNA', start="2021-05-24", end="2024-01-25")
print(ticker)
GetMRNAInfo = yf.Ticker('MRNA')
'''
print("Company Sector: ",GetMRNAInfo.info['sector'])
# print("Price Earnings Ratio: ", GetMRNAInfo.info['trailingPE'])
print("Company Beta: ", GetMRNAInfo.info['beta'])
'''
for key, value in GetMRNAInfo.info.items():
    print(key, ":", value)


from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MRNAdata = yf.download("MRNA", period='730d', interval='60m')