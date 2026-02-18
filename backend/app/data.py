from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

DataSource = Literal["auto", "local", "yfinance"]

DEFAULT_LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class PreparedDataset:
    ticker: str
    source: DataSource
    series: pd.Series
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    mean: float
    std: float
    transform: str


def normalize_indian_ticker(raw_ticker: str) -> str:
    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if ticker.startswith("^"):
        return ticker
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return ticker
    if "." in ticker:
        return ticker
    return f"{ticker}.NS"


def local_symbol_from_ticker(raw_ticker: str) -> str:
    normalized = normalize_indian_ticker(raw_ticker)
    if normalized.startswith("^"):
        return normalized[1:].upper()
    return normalized.split(".")[0].upper()


def list_available_symbols(local_data_dir: str | None = None) -> list[str]:
    data_dir = Path(local_data_dir) if local_data_dir else DEFAULT_LOCAL_DATA_DIR
    if not data_dir.exists():
        return []
    symbols = sorted({p.stem.upper() for p in data_dir.glob("*.csv")})
    return symbols


def _extract_close_series_from_df(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [col for col in df.columns if str(col[0]).strip().lower() == "close"]
        if not close_cols:
            raise ValueError("No Close column found in multi-index CSV.")
        close = pd.to_numeric(df[close_cols[0]], errors="coerce")

        date_cols = [
            col for col in df.columns if str(col[0]).strip().lower() in {"date", "price"}
        ]
        if date_cols:
            dates = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            dates = pd.to_datetime(df.index, errors="coerce")
    else:
        close_col = next((c for c in df.columns if str(c).strip().lower() == "close"), None)
        if close_col is None:
            raise ValueError("No Close column found in CSV.")
        close = pd.to_numeric(df[close_col], errors="coerce")

        date_col = next(
            (c for c in df.columns if str(c).strip().lower() in {"date", "price"}),
            None,
        )
        if date_col is None:
            date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col], errors="coerce")

    series = pd.Series(close.values, index=dates)
    series = series.dropna().astype(float)
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def _load_close_from_local_csv(csv_path: Path) -> pd.Series:
    attempts = [
        {"skiprows": [1, 2]},
        {},
        {"header": [0, 1], "skiprows": [2]},
    ]
    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            series = _extract_close_series_from_df(df)
            if len(series) >= 2:
                return series
        except Exception as exc:  # noqa: PERF203
            last_error = exc

    raise ValueError(f"Could not parse Close prices from '{csv_path}': {last_error}")


def _fetch_close_from_local(raw_ticker: str, local_data_dir: str | None = None) -> pd.Series:
    symbol = local_symbol_from_ticker(raw_ticker)
    data_dir = Path(local_data_dir) if local_data_dir else DEFAULT_LOCAL_DATA_DIR
    candidates = [
        data_dir / f"{symbol}.csv",
        data_dir / f"{normalize_indian_ticker(raw_ticker)}.csv",
    ]
    for path in candidates:
        if path.exists():
            return _load_close_from_local_csv(path)
    raise FileNotFoundError(f"No local CSV found for ticker '{symbol}' in '{data_dir}'.")


def _fetch_close_from_yfinance(raw_ticker: str, period: str, interval: str) -> tuple[str, pd.Series]:
    ticker = normalize_indian_ticker(raw_ticker)
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return ticker, close


def fetch_close_series(
    raw_ticker: str,
    period: str = "5y",
    interval: str = "1d",
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> tuple[str, pd.Series, DataSource]:
    ticker = normalize_indian_ticker(raw_ticker)

    if data_source in {"auto", "local"}:
        try:
            local_series = _fetch_close_from_local(raw_ticker=ticker, local_data_dir=local_data_dir)
            if len(local_series) < 80:
                raise ValueError(
                    f"Local data for '{ticker}' has only {len(local_series)} rows, need >= 80."
                )
            return ticker, local_series, "local"
        except Exception:
            if data_source == "local":
                raise

    yf_ticker, yf_series = _fetch_close_from_yfinance(
        raw_ticker=ticker,
        period=period,
        interval=interval,
    )
    if len(yf_series) < 80:
        raise ValueError(f"Not enough historical rows for '{yf_ticker}'. Got {len(yf_series)} rows.")
    return yf_ticker, yf_series, "yfinance"


def close_to_log_returns(close_values: np.ndarray) -> np.ndarray:
    close_values = close_values.astype(np.float64)
    if np.any(close_values <= 0):
        raise ValueError("Close prices must be positive for log-return transform.")
    returns = np.diff(np.log(close_values))
    return returns.astype(np.float32)


def build_windows(values: np.ndarray, input_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    total_window = input_len + pred_len
    if len(values) <= total_window:
        raise ValueError(f"Need more than {total_window} rows, but received {len(values)} rows.")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for start in range(0, len(values) - total_window + 1):
        x_list.append(values[start : start + input_len])
        y_list.append(values[start + input_len : start + total_window])

    x_arr = np.array(x_list, dtype=np.float32)[..., None]
    y_arr = np.array(y_list, dtype=np.float32)[..., None]
    return x_arr, y_arr


def prepare_dataset(
    raw_ticker: str,
    period: str,
    input_len: int,
    pred_len: int,
    val_ratio: float = 0.2,
    data_source: DataSource = "auto",
    local_data_dir: str | None = None,
) -> PreparedDataset:
    ticker, close_series, source = fetch_close_series(
        raw_ticker=raw_ticker,
        period=period,
        data_source=data_source,
        local_data_dir=local_data_dir,
    )

    close_values = close_series.values.astype(np.float32)
    log_returns = close_to_log_returns(close_values)
    x_all, y_all = build_windows(values=log_returns, input_len=input_len, pred_len=pred_len)

    split_idx = int(len(x_all) * (1 - val_ratio))
    split_idx = max(1, min(split_idx, len(x_all) - 1))

    x_train = x_all[:split_idx]
    y_train = y_all[:split_idx]
    x_val = x_all[split_idx:]
    y_val = y_all[split_idx:]

    mean = float(np.mean(x_train))
    std = float(np.std(x_train))
    if std < 1e-8:
        std = 1.0

    x_train = (x_train - mean) / std
    y_train = (y_train - mean) / std
    x_val = (x_val - mean) / std
    y_val = (y_val - mean) / std

    return PreparedDataset(
        ticker=ticker,
        source=source,
        series=close_series,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        mean=mean,
        std=std,
        transform="log_return",
    )


def make_future_business_days(last_date: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    current = pd.Timestamp(last_date)
    while len(dates) < horizon:
        current = current + pd.Timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates