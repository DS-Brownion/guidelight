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