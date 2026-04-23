/**
 * 与后端 JSON 契约一致的 fetch 封装（C2）。
 * 类型来自 `openapi.json` → `npm run gen:api` 生成的 `generated.d.ts`。
 */
import type { components } from "./generated";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type MerchantPredictRequest = components["schemas"]["MerchantPredictRequest"];
export type MerchantPredictResponse = components["schemas"]["MerchantPredictResponse"];
export type SearchRequest = components["schemas"]["SearchRequest"];
export type SearchResponse = components["schemas"]["SearchResponse"];
export type StatesResponse = components["schemas"]["StatesResponse"];

function apiBase(): string {
  const b = import.meta.env.VITE_API_BASE_URL;
  return typeof b === "string" ? b.replace(/\/$/, "") : "";
}

async function parseError(res: Response): Promise<string> {
  const t = await res.text();
  try {
    const j = JSON.parse(t) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return JSON.stringify(j.detail);
  } catch {
    /* ignore */
  }
  return t || res.statusText;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${apiBase()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return json<HealthResponse>("/api/health", { method: "GET" });
}

export function getStates(): Promise<StatesResponse> {
  return json<StatesResponse>("/api/v1/states", { method: "GET" });
}

export function postMerchantPredict(
  body: MerchantPredictRequest
): Promise<MerchantPredictResponse> {
  return json<MerchantPredictResponse>("/api/v1/merchant/predict", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function postSearch(body: SearchRequest): Promise<SearchResponse> {
  return json<SearchResponse>("/api/v1/search", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
