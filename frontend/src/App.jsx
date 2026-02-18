import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getHistory, getSymbols, predictStock, trainModel } from "./api";

const sourceOptions = ["local", "auto", "yfinance"];
const periodOptions = ["1y", "3y", "5y", "10y"];
const horizonPresets = [5, 10, 15, 30];

const initialTrain = {
  ticker: "RELIANCE",
  period: "5y",
  input_len: 60,
  pred_len: 5,
  epochs: 30,
  batch_size: 32,
  learning_rate: 0.001,
  data_source: "local",
  local_data_dir: "",
};

const initialPredict = {
  ticker: "RELIANCE",
  horizon: 10,
  history_points: 90,
  data_source: "local",
  local_data_dir: "",
};

function formatInr(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(value);
}

function formatPct(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)}%`;
}

export default function App() {
  const [trainForm, setTrainForm] = useState(initialTrain);
  const [predictForm, setPredictForm] = useState(initialPredict);
  const [symbols, setSymbols] = useState([]);

  const [trainResult, setTrainResult] = useState(null);
  const [forecastResult, setForecastResult] = useState(null);
  const [historyOnly, setHistoryOnly] = useState([]);

  const [loadingTrain, setLoadingTrain] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingSymbols, setLoadingSymbols] = useState(false);

  const [error, setError] = useState("");

  const todayLabel = useMemo(
    () =>
      new Intl.DateTimeFormat("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      }).format(new Date()),
    []
  );

  async function loadSymbols(dataSource, localDataDir) {
    if (dataSource === "yfinance") {
      setSymbols([]);
      return;
    }

    setLoadingSymbols(true);
    try {
      const data = await getSymbols({
        data_source: dataSource,
        local_data_dir: localDataDir || undefined,
      });
      setSymbols(data.symbols || []);
    } catch {
      setSymbols([]);
    } finally {
      setLoadingSymbols(false);
    }
  }

  useEffect(() => {
    loadSymbols(trainForm.data_source, trainForm.local_data_dir);
  }, [trainForm.data_source, trainForm.local_data_dir]);

  const chartData = useMemo(() => {
    if (forecastResult) {
      const history = forecastResult.history.map((point) => ({
        date: point.date,
        history: point.value,
        forecast: null,
      }));

      const forecast = forecastResult.forecast.map((point) => ({
        date: point.date,
        history: null,
        forecast: point.value,
      }));

      return [...history, ...forecast];
    }

    if (historyOnly.length > 0) {
      return historyOnly.map((point) => ({
        date: point.date,
        history: point.value,
        forecast: null,
      }));
    }

    return [];
  }, [forecastResult, historyOnly]);

  const forecastRows = useMemo(() => {
    if (!forecastResult) {
      return [];
    }

    const base = forecastResult.last_close;

    return forecastResult.forecast.map((point, index) => {
      const delta = point.value - base;
      const pct = base === 0 ? 0 : (delta / base) * 100;
      return {
        step: index + 1,
        date: point.date,
        value: point.value,
        delta,
        pct,
      };
    });
  }, [forecastResult]);

  const forecastStartDate = forecastResult?.forecast?.[0]?.date || null;
  const forecastEnd = forecastResult?.forecast?.at(-1)?.value;
  const lastClose = forecastResult?.last_close;
  const projectedChangePct =
    typeof forecastEnd === "number" && typeof lastClose === "number" && lastClose !== 0
      ? ((forecastEnd - lastClose) / lastClose) * 100
      : null;

  const kpis = [
    {
      title: "Data Universe",
      value: loadingSymbols ? "Loading" : `${symbols.length}`,
      note: trainForm.data_source === "yfinance" ? "Live source mode" : "Local symbols found",
      tone: "blue",
    },
    {
      title: "Direction Accuracy",
      value:
        trainResult && typeof trainResult.direction_accuracy === "number"
          ? formatPct(trainResult.direction_accuracy * 100)
          : "--",
      note: "Validation on first-step move",
      tone: "teal",
    },
    {
      title: "Last Close",
      value: typeof lastClose === "number" ? `Rs ${formatInr(lastClose)}` : "--",
      note: "From latest history point",
      tone: "slate",
    },
    {
      title: "Projected Change",
      value: projectedChangePct !== null ? formatPct(projectedChangePct) : "--",
      note: "End of forecast horizon",
      tone: projectedChangePct !== null && projectedChangePct >= 0 ? "green" : "amber",
    },
  ];

  function applySymbol(symbol) {
    setTrainForm((prev) => ({ ...prev, ticker: symbol }));
    setPredictForm((prev) => ({ ...prev, ticker: symbol }));
  }

  async function onTrainSubmit(event) {
    event.preventDefault();
    setError("");
    setLoadingTrain(true);

    try {
      const payload = {
        ...trainForm,
        input_len: Number(trainForm.input_len),
        pred_len: Number(trainForm.pred_len),
        epochs: Number(trainForm.epochs),
        batch_size: Number(trainForm.batch_size),
        learning_rate: Number(trainForm.learning_rate),
        local_data_dir: trainForm.local_data_dir || null,
      };

      const data = await trainModel(payload);
      setTrainResult(data);
      setPredictForm((prev) => ({
        ...prev,
        ticker: trainForm.ticker,
        data_source: trainForm.data_source,
        local_data_dir: trainForm.local_data_dir,
      }));
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Training failed");
    } finally {
      setLoadingTrain(false);
    }
  }

  async function onPredictSubmit(event) {
    event.preventDefault();
    setError("");
    setLoadingPredict(true);

    try {
      const payload = {
        ...predictForm,
        horizon: Number(predictForm.horizon),
        history_points: Number(predictForm.history_points),
        local_data_dir: predictForm.local_data_dir || null,
      };

      const data = await predictStock(payload);
      setForecastResult(data);
      setHistoryOnly([]);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Prediction failed");
    } finally {
      setLoadingPredict(false);
    }
  }

  async function onLoadHistory(event) {
    event.preventDefault();
    setError("");
    setLoadingHistory(true);

    try {
      const data = await getHistory({
        ticker: predictForm.ticker,
        history_points: Number(predictForm.history_points),
        data_source: predictForm.data_source,
        local_data_dir: predictForm.local_data_dir || undefined,
      });
      setHistoryOnly(data.history || []);
      setForecastResult(null);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "History fetch failed");
    } finally {
      setLoadingHistory(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <aside className="sidebar reveal delay-1">
        <div className="brand-block">
          <div className="brand-mark">K</div>
          <div>
            <h1>Kavout</h1>
            <p>Indian Stock Forecasting</p>
          </div>
        </div>

        <div className="sidebar-group">
          <h2>Workspace</h2>
          <div className="nav-chip active">Forecast Dashboard</div>
          {/* <div className="nav-chip">Model Monitoring</div>
          <div className="nav-chip">Data Pipeline</div> */}
        </div>

        <div className="sidebar-group">
          <h2>Quick Symbols</h2>
          {loadingSymbols ? <p className="muted">Loading symbols...</p> : null}
          <div className="symbol-grid">
            {symbols.slice(0, 16).map((symbol) => (
              <button key={symbol} type="button" className="symbol-chip" onClick={() => applySymbol(symbol)}>
                {symbol}
              </button>
            ))}
            {symbols.length === 0 && <p className="muted">No local symbols available.</p>}
          </div>
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <span>Market: Live</span>   {/* prevoius its api connecteed */}
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar reveal delay-2">
          <div>
            <h2>Forecast Operations Console</h2>
            <p>Run training jobs, validate model quality, and generate market forecasts.</p>
          </div>
          <div className="topbar-meta">
            <div className="meta-card">
              <span className="meta-label">Date</span>
              <span className="meta-value">{todayLabel}</span>
            </div>
            <div className="meta-card">
              <span className="meta-label">Ticker</span>
              <span className="meta-value">{predictForm.ticker.toUpperCase()}</span>
            </div>
          </div>
        </header>

        <section className="kpi-grid reveal delay-3">
          {kpis.map((kpi) => (
            <article key={kpi.title} className={`kpi-card ${kpi.tone}`}>
              <p className="kpi-title">{kpi.title}</p>
              <p className="kpi-value">{kpi.value}</p>
              <p className="kpi-note">{kpi.note}</p>
            </article>
          ))}
        </section>

        {error && (
          <div className="alert reveal delay-2">
            <strong>Request Error</strong>
            <span>{error}</span>
          </div>
        )}

        <section className="panel-grid">
          <article className="panel reveal delay-2">
            <div className="panel-head">
              <h3>Model Training</h3>
              <span className="pill">Train</span>
            </div>

            <form className="form" onSubmit={onTrainSubmit}>
              <div className="field-grid two-col">
                <label>
                  <span>Ticker</span>
                  <input
                    value={trainForm.ticker}
                    onChange={(e) => setTrainForm((s) => ({ ...s, ticker: e.target.value.toUpperCase() }))}
                    list="symbol-options"
                    placeholder="RELIANCE or ^NSEI"
                    required
                  />
                </label>

                <label>
                  <span>Data Source</span>
                  <select
                    value={trainForm.data_source}
                    onChange={(e) => setTrainForm((s) => ({ ...s, data_source: e.target.value }))}
                  >
                    {sourceOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="field-grid two-col">
                <label>
                  <span>Period</span>
                  <select
                    value={trainForm.period}
                    onChange={(e) => setTrainForm((s) => ({ ...s, period: e.target.value }))}
                  >
                    {periodOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>

                <label>
                  <span>Local Data Directory</span>
                  <input
                    value={trainForm.local_data_dir}
                    onChange={(e) => setTrainForm((s) => ({ ...s, local_data_dir: e.target.value }))}
                    placeholder="backend/data"
                  />
                </label>
              </div>

              <div className="field-grid three-col">
                <label>
                  <span>Input Length</span>
                  <input
                    type="number"
                    min="20"
                    max="512"
                    value={trainForm.input_len}
                    onChange={(e) => setTrainForm((s) => ({ ...s, input_len: e.target.value }))}
                  />
                </label>

                <label>
                  <span>Prediction Length</span>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={trainForm.pred_len}
                    onChange={(e) => setTrainForm((s) => ({ ...s, pred_len: e.target.value }))}
                  />
                </label>

                <label>
                  <span>Epochs</span>
                  <input
                    type="number"
                    min="1"
                    max="500"
                    value={trainForm.epochs}
                    onChange={(e) => setTrainForm((s) => ({ ...s, epochs: e.target.value }))}
                  />
                </label>
              </div>

              <div className="field-grid two-col">
                <label>
                  <span>Batch Size</span>
                  <input
                    type="number"
                    min="8"
                    max="512"
                    value={trainForm.batch_size}
                    onChange={(e) => setTrainForm((s) => ({ ...s, batch_size: e.target.value }))}
                  />
                </label>

                <label>
                  <span>Learning Rate</span>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.1"
                    value={trainForm.learning_rate}
                    onChange={(e) => setTrainForm((s) => ({ ...s, learning_rate: e.target.value }))}
                  />
                </label>
              </div>

              <button className="btn primary" type="submit" disabled={loadingTrain}>
                {loadingTrain ? "Training in progress..." : "Train Model"}
              </button>
            </form>

            <div className="result-card">
              <h4>Latest Training Snapshot</h4>
              {!trainResult ? (
                <p className="muted">No training run yet.</p>
              ) : (
                <ul>
                  <li>Ticker: {trainResult.ticker}</li>
                  <li>Source: {trainResult.source}</li>
                  <li>Validation Loss: {trainResult.val_loss.toFixed(6)}</li>
                  <li>Validation RMSE: {trainResult.val_rmse.toFixed(6)}</li>
                  <li>Direction Accuracy: {formatPct(trainResult.direction_accuracy * 100)}</li>
                </ul>
              )}
            </div>
          </article>

          <article className="panel reveal delay-3">
            <div className="panel-head">
              <h3>Forecast Execution</h3>
              <span className="pill alt">Predict</span>
            </div>

            <form className="form" onSubmit={onPredictSubmit}>
              <div className="field-grid two-col">
                <label>
                  <span>Ticker</span>
                  <input
                    value={predictForm.ticker}
                    onChange={(e) => setPredictForm((s) => ({ ...s, ticker: e.target.value.toUpperCase() }))}
                    list="symbol-options"
                    required
                  />
                </label>

                <label>
                  <span>Data Source</span>
                  <select
                    value={predictForm.data_source}
                    onChange={(e) => setPredictForm((s) => ({ ...s, data_source: e.target.value }))}
                  >
                    {sourceOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="field-grid two-col">
                <label>
                  <span>Horizon (Days)</span>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    value={predictForm.horizon}
                    onChange={(e) => setPredictForm((s) => ({ ...s, horizon: e.target.value }))}
                  />
                </label>

                <label>
                  <span>History Points</span>
                  <input
                    type="number"
                    min="20"
                    max="500"
                    value={predictForm.history_points}
                    onChange={(e) => setPredictForm((s) => ({ ...s, history_points: e.target.value }))}
                  />
                </label>
              </div>

              <label>
                <span>Local Data Directory</span>
                <input
                  value={predictForm.local_data_dir}
                  onChange={(e) => setPredictForm((s) => ({ ...s, local_data_dir: e.target.value }))}
                  placeholder="backend/data"
                />
              </label>

              <div className="preset-row">
                {horizonPresets.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    className={`preset ${Number(predictForm.horizon) === preset ? "active" : ""}`}
                    onClick={() => setPredictForm((s) => ({ ...s, horizon: preset }))}
                  >
                    {preset}D
                  </button>
                ))}
              </div>

              <div className="action-row">
                <button className="btn ghost" type="button" disabled={loadingHistory} onClick={onLoadHistory}>
                  {loadingHistory ? "Loading history..." : "Load History"}
                </button>
                <button className="btn primary" type="submit" disabled={loadingPredict}>
                  {loadingPredict ? "Running forecast..." : "Run Forecast"}
                </button>
              </div>
            </form>

            <div className="result-card">
              <h4>Latest Forecast Snapshot</h4>
              {!forecastResult ? (
                <p className="muted">No forecast generated yet.</p>
              ) : (
                <ul>
                  <li>Ticker: {forecastResult.ticker}</li>
                  <li>Source: {forecastResult.source}</li>
                  <li>Last Close: Rs {formatInr(forecastResult.last_close)}</li>
                  <li>Horizon: {forecastResult.horizon} days</li>
                  <li>Projected Change: {projectedChangePct !== null ? formatPct(projectedChangePct) : "--"}</li>
                </ul>
              )}
            </div>
          </article>
        </section>

        <section className="panel chart-panel reveal delay-3">
          <div className="panel-head">
            <h3>Price Trajectory</h3>
            <span className="muted">History and forecast overlay</span>
          </div>

          {chartData.length === 0 ? (
            <div className="empty">Load history or run forecast to populate the chart.</div>
          ) : (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={390}>
                <LineChart data={chartData} margin={{ top: 10, right: 20, left: 8, bottom: 8 }}>
                  <defs>
                    <linearGradient id="histLine" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#0f7aed" stopOpacity="1" />
                      <stop offset="100%" stopColor="#0f7aed" stopOpacity="0.4" />
                    </linearGradient>
                    <linearGradient id="fcLine" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef9b0f" stopOpacity="1" />
                      <stop offset="100%" stopColor="#ef9b0f" stopOpacity="0.5" />
                    </linearGradient>
                  </defs>

                  <CartesianGrid strokeDasharray="4 4" stroke="#d5e1ed" />
                  <XAxis dataKey="date" minTickGap={24} />
                  <YAxis width={84} tickFormatter={(value) => `Rs ${Math.round(value)}`} />
                  <Tooltip
                    formatter={(value) => [`Rs ${formatInr(Number(value))}`, "Price"]}
                    labelStyle={{ color: "#0f1f2e", fontWeight: 700 }}
                  />
                  <Legend />

                  {forecastStartDate && (
                    <ReferenceArea
                      x1={forecastStartDate}
                      x2={chartData.at(-1)?.date}
                      fill="#0f7aed"
                      fillOpacity={0.07}
                    />
                  )}

                  <Line
                    type="monotone"
                    dataKey="history"
                    name="History"
                    stroke="url(#histLine)"
                    dot={false}
                    strokeWidth={2.5}
                    activeDot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast"
                    stroke="url(#fcLine)"
                    dot={false}
                    strokeWidth={3}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="panel table-panel reveal delay-4">
          <div className="panel-head">
            <h3>Forecast Tape</h3>
            <span className="muted">Stepwise projected values vs last close</span>
          </div>

          {forecastRows.length === 0 ? (
            <div className="empty">Run forecast to view stepwise projections.</div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Step</th>
                    <th>Date</th>
                    <th>Forecast (Rs)</th>
                    <th>Delta (Rs)</th>
                    <th>Delta %</th>
                  </tr>
                </thead>
                <tbody>
                  {forecastRows.map((row) => (
                    <tr key={row.date}>
                      <td>{row.step}</td>
                      <td>{row.date}</td>
                      <td>{formatInr(row.value)}</td>
                      <td className={row.delta >= 0 ? "positive" : "negative"}>
                        {row.delta >= 0 ? "+" : ""}
                        {formatInr(row.delta)}
                      </td>
                      <td className={row.pct >= 0 ? "positive" : "negative"}>
                        {row.pct >= 0 ? "+" : ""}
                        {formatPct(row.pct)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <datalist id="symbol-options">
        {symbols.map((symbol) => (
          <option key={symbol} value={symbol} />
        ))}
      </datalist>
    </div>
  );
}