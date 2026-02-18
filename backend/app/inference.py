from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .data import close_to_log_returns, fetch_close_series, make_future_business_days, normalize_indian_ticker
from .model import Model
from .schemas import DataSource

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def sanitize_ticker_for_filename(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace(".", "_").replace("/", "_").replace(" ", "_")


def find_latest_artifact_for_ticker(raw_ticker: str) -> Path:
    normalized = normalize_indian_ticker(raw_ticker)
    safe = sanitize_ticker_for_filename(normalized)
    matches = sorted(
        ARTIFACT_DIR.glob(f"{safe}_in*_out*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No trained model found for '{normalized}'. Train first.")
    return matches[0]


def iterative_forecast(model: Model, normalized_window: np.ndarray, input_len: int, horizon: int, device: torch.device) -> np.ndarray:
    preds: list[float] = []
    rolling = normalized_window.copy().astype(np.float32)

    while len(preds) < horizon:
        with torch.no_grad():
            x = torch.tensor(rolling, dtype=torch.float32, device=device).view(1, input_len, 1)
            out = model(x).cpu().numpy().reshape(-1)

        take = min(len(out), horizon - len(preds))
        chunk = out[:take]
        preds.extend(chunk.tolist())

        rolling = np.concatenate([rolling[take:], chunk], axis=0)
        if len(rolling) > input_len:
            rolling = rolling[-input_len:]

    return np.array(preds, dtype=np.float32)


def get_history_for_ticker(
    raw_ticker: str,
    history_points: int,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
    period: str = "5y",
) -> dict:
    ticker, series, used_source = fetch_close_series(
        raw_ticker=raw_ticker,
        period=period,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )
    history_series = series.tail(history_points)
    history = [
        {"date": idx.strftime("%Y-%m-%d"), "value": float(value)}
        for idx, value in history_series.items()
    ]
    return {
        "ticker": ticker,
        "source": used_source,
        "history": history,
    }


def _predict_from_log_return_model(
    model: Model,
    series: np.ndarray,
    input_len: int,
    mean: float,
    std: float,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    if len(series) < input_len + 1:
        raise ValueError(f"Need at least {input_len + 1} close points, got {len(series)}.")

    recent_close = series[-(input_len + 1) :]
    log_return_window = close_to_log_returns(recent_close)
    normalized_window = (log_return_window - mean) / std

    normalized_pred = iterative_forecast(
        model=model,
        normalized_window=normalized_window,
        input_len=input_len,
        horizon=horizon,
        device=device,
    )
    pred_log_returns = normalized_pred * std + mean

    last_close = float(series[-1])
    cumulative_returns = np.cumsum(pred_log_returns)
    forecast_prices = last_close * np.exp(cumulative_returns)
    return forecast_prices.astype(np.float32)


def _predict_from_close_level_model(
    model: Model,
    series: np.ndarray,
    input_len: int,
    mean: float,
    std: float,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    if len(series) < input_len:
        raise ValueError(f"Need at least {input_len} close points, got {len(series)}.")

    normalized_window = (series[-input_len:] - mean) / std
    normalized_pred = iterative_forecast(
        model=model,
        normalized_window=normalized_window,
        input_len=input_len,
        horizon=horizon,
        device=device,
    )
    return (normalized_pred * std + mean).astype(np.float32)


def predict_with_saved_model(
    raw_ticker: str,
    horizon: int,
    history_points: int,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> dict:
    artifact_path = find_latest_artifact_for_ticker(raw_ticker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(artifact_path, map_location=device)
    input_len = int(checkpoint["input_len"])
    pred_len = int(checkpoint["pred_len"])

    config = SimpleNamespace(seq_len=input_len, pred_len=pred_len, individual=False, enc_in=1)
    model = Model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    ticker = checkpoint["ticker"]
    period = checkpoint.get("period", "5y")
    transform = checkpoint.get("transform", "close_zscore")
    preferred_source: DataSource = data_source
    if preferred_source == "auto":
        preferred_source = checkpoint.get("source", "auto")

    mean = float(checkpoint["mean"])
    std = float(checkpoint["std"])
    if std < 1e-8:
        std = 1.0

    _, series, used_source = fetch_close_series(
        raw_ticker=ticker,
        period=period,
        data_source=preferred_source,
        local_data_dir=local_data_dir,
    )

    close_values = series.values.astype(np.float32)

    if transform == "log_return":
        forecast_values = _predict_from_log_return_model(
            model=model,
            series=close_values,
            input_len=input_len,
            mean=mean,
            std=std,
            horizon=horizon,
            device=device,
        )
    else:
        forecast_values = _predict_from_close_level_model(
            model=model,
            series=close_values,
            input_len=input_len,
            mean=mean,
            std=std,
            horizon=horizon,
            device=device,
        )

    history_series = series.tail(history_points)
    future_dates = make_future_business_days(series.index[-1], horizon)

    history = [
        {"date": idx.strftime("%Y-%m-%d"), "value": float(value)}
        for idx, value in history_series.items()
    ]
    forecast = [
        {"date": dt.strftime("%Y-%m-%d"), "value": float(value)}
        for dt, value in zip(future_dates, forecast_values, strict=True)
    ]

    return {
        "ticker": ticker,
        "source": used_source,
        "transform": transform,
        "model_artifact": str(artifact_path),
        "input_len": input_len,
        "pred_len": pred_len,
        "horizon": horizon,
        "last_close": float(series.iloc[-1]),
        "history": history,
        "forecast": forecast,
    }