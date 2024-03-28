# defines the request bodies for the API
from pydantic import BaseModel

# request body for the api to run inference on the given model

class OptimizationRequest(BaseModel):
    portfolio_name: str
    tickers: list[str]
    weights: list[float]
    risk_free_rate: float
    expected_return: float
    expected_volatility: float

class InferenceRequest(BaseModel):
    ticker: str
    description: str | None = None
    price: float
    tax: float | None = None