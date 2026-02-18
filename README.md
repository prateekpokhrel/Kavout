# DLinear Indian Stock Forecast (Full Stack)

Full-stack project for forecasting Indian stock market prices with DLinear.

- Backend: FastAPI + PyTorch
- Frontend: React (Vite)
- Model: DLinear architecture from your provided code

## What Was Fixed

1. Training and inference now use the same preprocessing transform.
2. Forecasting uses log-returns internally, then converts back to price levels.
3. Added local CSV support (`backend/data`) with yfinance fallback.
4. Added symbol/history endpoints and quality metrics (`val_rmse`, direction accuracy).

## Project Structure

- `backend/app/model.py`: DLinear model
- `backend/app/data.py`: data loading + preprocessing
- `backend/app/trainer.py`: training + artifact save
- `backend/app/inference.py`: artifact loading + forecasting
- `backend/app/main.py`: API routes
- `frontend/src/App.jsx`: UI for train/predict/history

## Backend Run

```bash
cd stock-forecast-dlinear/backend
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend Run

```bash
cd stock-forecast-dlinear/frontend
npm install
copy .env.example .env
npm run dev
```

## API

- `GET /health`
- `GET /api/symbols?data_source=local`
- `GET /api/history?ticker=RELIANCE&history_points=120&data_source=local`
- `POST /api/train`
- `POST /api/predict`

### Train Example

```bash
curl -X POST http://127.0.0.1:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"ticker":"RELIANCE","period":"5y","input_len":60,"pred_len":5,"epochs":30,"data_source":"local"}'
```

### Predict Example

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker":"RELIANCE","horizon":10,"history_points":90,"data_source":"local"}'
```

## Note

This project is for educational use and not financial advice.