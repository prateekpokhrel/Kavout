from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .data import list_available_symbols
from .inference import get_history_for_ticker, predict_with_saved_model
from .schemas import (
    DataSource,
    HistoryResponse,
    PredictRequest,
    PredictResponse,
    SymbolsResponse,
    TrainRequest,
    TrainResponse,
)
from .trainer import train_and_save_model

app = FastAPI(
    title="Indian Stock DLinear API",
    version="2.0.0",
    description="Train and forecast Indian stocks using DLinear with consistent preprocessing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/symbols", response_model=SymbolsResponse)
def symbols_endpoint(
    data_source: DataSource = Query(default="auto"),
    local_data_dir: str | None = Query(default=None),
) -> SymbolsResponse:
    if data_source == "yfinance":
        return SymbolsResponse(source="yfinance", symbols=[])

    symbols = list_available_symbols(local_data_dir=local_data_dir)
    return SymbolsResponse(source="local", symbols=symbols)


@app.get("/api/history", response_model=HistoryResponse)
def history_endpoint(
    ticker: str = Query(...),
    history_points: int = Query(default=120, ge=20, le=500),
    period: str = Query(default="5y"),
    data_source: DataSource = Query(default="auto"),
    local_data_dir: str | None = Query(default=None),
) -> HistoryResponse:
    try:
        result = get_history_for_ticker(
            raw_ticker=ticker,
            history_points=history_points,
            data_source=data_source,
            local_data_dir=local_data_dir,
            period=period,
        )
        return HistoryResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"History failed: {exc}") from exc


@app.post("/api/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest) -> TrainResponse:
    try:
        result = train_and_save_model(req)
        return TrainResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc


@app.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    try:
        result = predict_with_saved_model(
            raw_ticker=req.ticker,
            horizon=req.horizon,
            history_points=req.history_points,
            data_source=req.data_source,
            local_data_dir=req.local_data_dir,
        )
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc