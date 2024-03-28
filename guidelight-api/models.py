from tortoise import fields
from tortoise.models import Model

class CompanyFinancials(Model):
    financial_id = fields.IntField(pk=True)
    ticker = fields.CharField(max_length=10)
    fiscal_year = fields.IntField()
    total_revenue = fields.BigIntField()
    net_income = fields.BigIntField()
    ebitda = fields.BigIntField()
    total_assets = fields.BigIntField()
    total_liabilities = fields.BigIntField()
    equity = fields.BigIntField()
    average_sentiment_score = fields.FloatField()
    news_ticker_list = fields.TextField()
    last_updated = fields.DatetimeField(auto_now=True)

class HourlyStockData(Model):
    data_id = fields.IntField(pk=True)
    ticker = fields.CharField(max_length=10)
    datetime = fields.DatetimeField()
    close_price = fields.FloatField()
    volume = fields.BigIntField()
    last_updated = fields.DatetimeField(auto_now=True)

class InsiderTrades(Model):
    trade_id = fields.IntField(pk=True)
    ticker = fields.CharField(max_length=10)
    date = fields.DateField()
    insider_name = fields.CharField(max_length=50)
    position = fields.CharField(max_length=50)
    transaction_type = fields.CharField(max_length=50)
    quantity = fields.BigIntField()
    price = fields.FloatField()
    last_updated = fields.DatetimeField(auto_now=True)


class Dividends(Model):
    dividend_id = fields.IntField(pk=True)
    ticker = fields.CharField(max_length=10)
    ex_dividendDate = fields.DateField()
    payment_date = fields.DateField()
    dividend_amount = fields.FloatField()
    last_updated = fields.DatetimeField(auto_now=True)
class MarketIndicators(Model):
    indicator_id = fields.IntField(pk=True)
    date = fields.DateField()
    sp500 = fields.FloatField()
    djia = fields.FloatField()
    nasdaq = fields.FloatField()
    vix = fields.FloatField()
    gold_price = fields.FloatField()
    oil_price = fields.FloatField()
    usd_treasury_yield = fields.FloatField()
    last_updated = fields.DatetimeField(auto_now=True)
