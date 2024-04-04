# defines the request bodies for the API
from pydantic import BaseModel

# request body for the api to run inference on the given model

class OptimizationRequest(BaseModel):
    """
    Represents a request body for the portfolio target for optimization.

    Attributes:
        portfolio_name (str): The name of the portfolio.
        tickers (list[str]): A list of targeted ticker symbols.
        weights (list[float]): A list of weights corresponding to the tickers.
        risk_free_rate (float): The risk-free rate for the optimization.
        expected_return (float): The expected return for the optimization.
        expected_volatility (float): The expected volatility for the optimization.
    """
    
    portfolio_name: str
    tickers: list[str]
    weights: list[float]
    risk_free_rate: float
    expected_return: float
    expected_volatility: float

class InferenceRequest(BaseModel):
    """
    Represents the body for price inference request.

    Attributes:
        ticker (str): The ticker symbol.
        description (str, optional): The description of the request. Defaults to None.
        price (float): The price of the request.
        tax (float, optional): The tax amount. Defaults to None.
    """

    ticker: str
    description: str | None = None
    price: float
    tax: float | None = None