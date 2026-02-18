from typing import Literal

from pydantic import BaseModel, Field

DataSource = Literal["auto", "local", "yfinance"]


class TrainRequest(BaseModel):
    ticker: str = Field(..., examples=["RELIANCE", "^NSEI"])
    period: str = Field(default="5y", examples=["1y", "3y", "5y", "10y"])
    input_len: int = Field(default=60, ge=20, le=512)
    pred_len: int = Field(default=5, ge=1, le=120)
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=0.001, gt=0, lt=1)
    data_source: DataSource = Field(default="auto")
    local_data_dir: str | None = Field(default=None)


class PredictRequest(BaseModel):
    ticker: str = Field(..., examples=["RELIANCE", "^NSEI"])
    horizon: int = Field(default=10, ge=1, le=120)
    history_points: int = Field(default=90, ge=20, le=500)
    data_source: DataSource = Field(default="auto")
    local_data_dir: str | None = Field(default=None)


class PricePoint(BaseModel):
    date: str
    value: float


class TrainResponse(BaseModel):
    ticker: str
    source: DataSource
    transform: str
    artifact_path: str
    train_loss: float
    val_loss: float
    val_rmse: float
    direction_accuracy: float
    input_len: int
    pred_len: int
    train_samples: int
    val_samples: int
    trained_at_utc: str


class PredictResponse(BaseModel):
    ticker: str
    source: DataSource
    transform: str
    model_artifact: str
    input_len: int
    pred_len: int
    horizon: int
    last_close: float
    history: list[PricePoint]
    forecast: list[PricePoint]


class SymbolsResponse(BaseModel):
    source: DataSource
    symbols: list[str]


class HistoryResponse(BaseModel):
    ticker: str
    source: DataSource
    history: list[PricePoint]