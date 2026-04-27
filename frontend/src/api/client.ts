/**
 * Fetch helpers aligned with the backend JSON contract.
 * Types are generated from `openapi.json` via `npm run gen:api` → `generated.d.ts`.
 */
import type { components } from "./generated";

export type HealthResponse = components["schemas"]["HealthResponse"];
export type MerchantPredictRequest = components["schemas"]["MerchantPredictRequest"];
export type MerchantPredictResponse = components["schemas"]["MerchantPredictResponse"];
export type MerchantCoverageResponse = components["schemas"]["MerchantCoverageResponse"];
export type MerchantCityRow = components["schemas"]["MerchantCityRow"];
export type MerchantCitiesResponse = components["schemas"]["MerchantCitiesResponse"];
export type MerchantCategoriesResponse = components["schemas"]["MerchantCategoriesResponse"];
export type StatesResponse = components["schemas"]["StatesResponse"];
type GeneratedSearchRequest = components["schemas"]["SearchRequest"];
type GeneratedSearchResponse = components["schemas"]["SearchResponse"];

export interface SearchActionEvent {
  action: "detail_open" | "like" | "pass" | "refresh" | "slider_override";
  business_id?: string | null;
  query_text?: string | null;
}

export type SearchRequest = GeneratedSearchRequest & {
  user_location?: string | null;
  rl_enabled?: boolean;
  rl_user_overrode?: boolean;
  rl_prev_selected_arm?: string | null;
  rl_prev_intent_name?: string | null;
  rl_action_events?: SearchActionEvent[];
};

export type SearchMeta = Record<string, unknown> & {
  rl_applied?: boolean;
  rl_intent_name?: string | null;
  rl_selected_arm?: string | null;
  rl_strategy_label?: string | null;
  rl_effective_weights?: Record<string, number> | null;
  rl_user_override_active?: boolean;
};

export type SearchResponse = Omit<GeneratedSearchResponse, "meta"> & {
  meta: SearchMeta;
};

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

export function getMerchantCities(params?: { min_rows?: number }): Promise<MerchantCitiesResponse> {
  const sp = new URLSearchParams();
  if (params?.min_rows != null) {
    sp.set("min_rows", String(params.min_rows));
  }
  const q = sp.toString();
  return json<MerchantCitiesResponse>(`/api/v1/merchant/cities${q ? `?${q}` : ""}`, { method: "GET" });
}

export function getMerchantCategories(params?: {
  city?: string | null;
  state?: string | null;
  max_rows_if_no_city?: number;
}): Promise<MerchantCategoriesResponse> {
  const sp = new URLSearchParams();
  if (params?.city != null && String(params.city).trim()) {
    sp.set("city", String(params.city).trim());
  }
  if (params?.state != null && String(params.state).trim()) {
    sp.set("state", String(params.state).trim().toUpperCase());
  }
  if (params?.max_rows_if_no_city != null) {
    sp.set("max_rows_if_no_city", String(params.max_rows_if_no_city));
  }
  const q = sp.toString();
  return json<MerchantCategoriesResponse>(`/api/v1/merchant/categories${q ? `?${q}` : ""}`, { method: "GET" });
}

/** Map free text (e.g. "burger", "fast food") to train_spatial ``cat_*`` for the current city slice. */
export function getMerchantCategoryResolve(params: {
  q: string;
  city?: string | null;
  state?: string | null;
  max_rows_if_no_city?: number;
}): Promise<MerchantCategoriesResponse> {
  const sp = new URLSearchParams();
  sp.set("q", params.q.trim());
  if (params.city != null && String(params.city).trim()) {
    sp.set("city", String(params.city).trim());
  }
  if (params.state != null && String(params.state).trim()) {
    sp.set("state", String(params.state).trim().toUpperCase());
  }
  if (params.max_rows_if_no_city != null) {
    sp.set("max_rows_if_no_city", String(params.max_rows_if_no_city));
  }
  return json<MerchantCategoriesResponse>(`/api/v1/merchant/categories/resolve?${sp.toString()}`, { method: "GET" });
}

export function getMerchantCoverage(params: {
  city?: string | null;
  state?: string | null;
  max_rows_if_no_city?: number;
  max_sample_points?: number;
}): Promise<MerchantCoverageResponse> {
  const sp = new URLSearchParams();
  if (params.city != null && String(params.city).trim()) {
    sp.set("city", String(params.city).trim());
  }
  if (params.state != null && String(params.state).trim()) {
    sp.set("state", String(params.state).trim().toUpperCase());
  }
  if (params.max_rows_if_no_city != null) {
    sp.set("max_rows_if_no_city", String(params.max_rows_if_no_city));
  }
  if (params.max_sample_points != null) {
    sp.set("max_sample_points", String(params.max_sample_points));
  }
  const q = sp.toString();
  return json<MerchantCoverageResponse>(`/api/v1/merchant/coverage${q ? `?${q}` : ""}`, {
    method: "GET",
  });
}

export function postSearch(body: SearchRequest): Promise<SearchResponse> {
  return json<SearchResponse>("/api/v1/search", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
