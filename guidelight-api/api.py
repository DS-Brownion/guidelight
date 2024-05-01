from fastapi import FastAPI
from models import *
from reqbody import *
import httpx as httpreq
app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


# inference enpoint
@app.get("/inference/price")
async def inference():
    """
    inference enpoint

    Returns:
        dict: A dictionary containing the description of the inference.
    """
    return {"description": "For making inferences on the stock ticker price"}
app.get("/inference/volatility")
async def inference():
    return {"description": "For making inferences on the stock ticker volatility"}
app.get("/inference/returns{ticker}")
async def returns(input: InferenceRequest, q:str):
    # makes inference on the given ticker returns after estimating the Heston model parameters given a timeframe
    async with httpreq.AsyncClient() as client:
        response = await client.post('https://external.api/inference', json=input.dict())
        return response.json()
# portfolio optimization endpoint
@app.get("/portfolio/optimization/{portfolio_name}")
async def portfolio_optimization(portfolio_name: str):
    pass
# stock summary endpoint
# stock aggregates endpoints
# stock financials endpoint
# stock sentiment endpoint
# stock headlines endpoint
# reccomendations output endpoint

# stock insider trades endpoint


"""
This script defines an API using FastAPI framework for making inferences on stock ticker price, volatility, and returns, as well as performing portfolio optimization.

Endpoints:
- "/" : Returns a simple greeting message.
- "/inference/price" : Returns a description of the inference endpoint for stock ticker price.
- "/inference/volatility" : Returns a description of the inference endpoint for stock ticker volatility.
- "/inference/returns{ticker}" : Makes an inference on the given ticker returns after estimating the Heston model parameters given a timeframe.
- "/portfolio/optimization/{portfolio_name}" : Performs portfolio optimization for the specified portfolio name.

To use this API, you can send HTTP GET requests to the respective endpoints to retrieve the desired information.

Note: The code provided is incomplete and requires additional implementation for the remaining endpoints.
"""