import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
});

export async function trainModel(payload) {
  const { data } = await api.post("/api/train", payload);
  return data;
}

export async function predictStock(payload) {
  const { data } = await api.post("/api/predict", payload);
  return data;
}

export async function getSymbols(params = {}) {
  const { data } = await api.get("/api/symbols", { params });
  return data;
}

export async function getHistory(params) {
  const { data } = await api.get("/api/history", { params });
  return data;
}