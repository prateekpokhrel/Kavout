from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare_dataset
from .model import Model
from .schemas import TrainRequest

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_ticker_for_filename(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace(".", "_").replace("/", "_").replace(" ", "_")


def compute_validation_metrics(model: Model, val_x: torch.Tensor, val_y: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        pred = model(val_x.to(device)).cpu().numpy()
        true = val_y.cpu().numpy()

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    pred_sign = np.sign(pred[:, 0, 0])
    true_sign = np.sign(true[:, 0, 0])
    direction_accuracy = float(np.mean(pred_sign == true_sign))
    return rmse, direction_accuracy


def train_and_save_model(req: TrainRequest) -> dict:
    dataset = prepare_dataset(
        raw_ticker=req.ticker,
        period=req.period,
        input_len=req.input_len,
        pred_len=req.pred_len,
        data_source=req.data_source,
        local_data_dir=req.local_data_dir,
    )

    config = SimpleNamespace(
        seq_len=req.input_len,
        pred_len=req.pred_len,
        individual=False,
        enc_in=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config).to(device)

    train_x = torch.tensor(dataset.x_train, dtype=torch.float32)
    train_y = torch.tensor(dataset.y_train, dtype=torch.float32)
    val_x = torch.tensor(dataset.x_val, dtype=torch.float32)
    val_y = torch.tensor(dataset.y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=req.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=req.batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=req.learning_rate)

    train_loss = 0.0
    val_loss = 0.0

    for _ in range(req.epochs):
        model.train()
        total_train = 0.0
        seen_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            size = batch_x.size(0)
            total_train += float(loss.item()) * size
            seen_train += size

        model.eval()
        total_val = 0.0
        seen_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                size = batch_x.size(0)
                total_val += float(loss.item()) * size
                seen_val += size

        train_loss = total_train / max(1, seen_train)
        val_loss = total_val / max(1, seen_val)

    val_rmse, direction_accuracy = compute_validation_metrics(
        model=model,
        val_x=val_x,
        val_y=val_y,
        device=device,
    )

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    safe_ticker = sanitize_ticker_for_filename(dataset.ticker)
    artifact_name = f"{safe_ticker}_in{req.input_len}_out{req.pred_len}.pt"
    artifact_path = ARTIFACT_DIR / artifact_name

    checkpoint = {
        "state_dict": model.state_dict(),
        "ticker": dataset.ticker,
        "source": dataset.source,
        "period": req.period,
        "input_len": req.input_len,
        "pred_len": req.pred_len,
        "transform": dataset.transform,
        "mean": dataset.mean,
        "std": dataset.std,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "direction_accuracy": direction_accuracy,
        "train_samples": int(dataset.x_train.shape[0]),
        "val_samples": int(dataset.x_val.shape[0]),
        "trained_at_utc": trained_at,
    }

    torch.save(checkpoint, artifact_path)

    return {
        "ticker": dataset.ticker,
        "source": dataset.source,
        "transform": dataset.transform,
        "artifact_path": str(artifact_path),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "direction_accuracy": direction_accuracy,
        "input_len": req.input_len,
        "pred_len": req.pred_len,
        "train_samples": int(dataset.x_train.shape[0]),
        "val_samples": int(dataset.x_val.shape[0]),
        "trained_at_utc": trained_at,
    }