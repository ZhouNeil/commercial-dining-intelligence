<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import type { GeoJsonObject } from "geojson";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  getMerchantCities,
  SPATIAL_CITY_MIN_TRAIN_ROWS,
  getMerchantCoverage,
  postMerchantHeatmap,
  postMerchantPredict,
  type MerchantCityRow,
  type MerchantCoverageResponse,
  type MerchantHeatmapRequest,
  type MerchantHeatmapResponse,
  type MerchantPredictResponse,
} from "../api/client";

const route = useRoute();
const router = useRouter();

const city = ref("Philadelphia");
const stateFilter = ref("");
const lat = ref(39.9526);
const lon = ref(-75.1652);
const maxRows = ref(2000);
/** Free text → backend ``resolve_merchant_category_text`` → ``cat_*`` (Yelp one-hot style). */
const businessTypeText = ref("fast food, coffee");
/** Yelp-style tier 1–4; optional per-person price below if unset */
const priceTier = ref<"" | "1" | "2" | "3" | "4">("");
/** Approx USD per person; maps to 1–4 when no tier selected */
const pricePerPerson = ref("");
const loading = ref(false);
/** Map result panel visibility (toolbar popover removed; only this panel + close). */
const resultDockOpen = ref(true);
const coverageLoading = ref(false);
const err = ref<string | null>(null);
const result = ref<MerchantPredictResponse | null>(null);
const coverage = ref<MerchantCoverageResponse | null>(null);
/** pin = click map for single-site prediction; heatmap = full-slice grid */
const viewTab = ref<"pin" | "heatmap">("pin");
const heatmapData = ref<MerchantHeatmapResponse | null>(null);
const heatmapLoading = ref(false);
const heatmapScoreMode = ref<"rule" | "ml" | "survival" | "stars">("rule");
const heatmapGridSize = ref(12);

const cities = ref<MerchantCityRow[]>([]);
const citySearch = ref("");

const mapEl = ref<HTMLElement | null>(null);
let map: L.Map | null = null;
let hullLayer: L.Layer | null = null;
let sampleLayer: L.Layer | null = null;
let pinLayer: L.LayerGroup | null = null;
let heatmapRectLayer: L.LayerGroup | null = null;

function formatCategoryLabel(col: string): string {
  const s = col.startsWith("cat_") ? col.slice(4) : col;
  return s.replace(/_/g, " ").replace(/\s+/g, " ").replace(/&/g, " & ");
}

const resolvedCategoryLine = computed(() => {
  const keys = result.value?.resolved_category_keys;
  if (!keys?.length) return null;
  return keys.map((k) => formatCategoryLabel(k)).join(" · ");
});

const heatmapResolvedLine = computed(() => {
  const keys = heatmapData.value?.resolved_category_keys;
  if (!keys?.length) return null;
  return keys.map((k) => formatCategoryLabel(k)).join(" · ");
});

const heatmapMetricTitle = computed(() => {
  if (heatmapScoreMode.value === "ml") return "ML score (0–100)";
  if (heatmapScoreMode.value === "survival") return "Survival P (0–1)";
  if (heatmapScoreMode.value === "stars") return "Predicted stars (0–5)";
  return "Rule-based score (0–100)";
});

const heatmapLegendEnds = computed(() => {
  if (heatmapScoreMode.value === "survival") return { lo: "~0", hi: "~1" };
  if (heatmapScoreMode.value === "stars") return { lo: "~1★", hi: "~5★" };
  return { lo: "Lower", hi: "Higher" };
});

function priceFitLabel(fit: string | null | undefined): string {
  if (fit === "good") return "Good";
  if (fit === "medium") return "Medium";
  if (fit === "poor") return "Poor";
  return "—";
}

const riskEntries = computed(() => {
  const r = result.value?.risk;
  if (!r || typeof r !== "object") return [];
  return Object.entries(r);
});

const filteredCities = computed(() => {
  const q = citySearch.value.trim().toLowerCase();
  if (!q) return cities.value;
  return cities.value.filter(
    (c) =>
      c.city.toLowerCase().includes(q) ||
      (c.state && c.state.toLowerCase().includes(q))
  );
});

function letterFor(name: string): string {
  const ch = name.trim().charAt(0).toUpperCase();
  return ch >= "A" && ch <= "Z" ? ch : "#";
}

const citiesByLetter = computed(() => {
  const m = new Map<string, MerchantCityRow[]>();
  for (const c of filteredCities.value) {
    const L0 = letterFor(c.city);
    if (!m.has(L0)) m.set(L0, []);
    m.get(L0)!.push(c);
  }
  const letters = [...m.keys()].sort((a, b) => {
    if (a === "#") return 1;
    if (b === "#") return -1;
    return a.localeCompare(b);
  });
  return { letters, groups: m };
});

let debounceTimer: ReturnType<typeof setTimeout> | null = null;

function destroyMap() {
  if (map) {
    map.remove();
    map = null;
  }
  hullLayer = null;
  sampleLayer = null;
  pinLayer = null;
  heatmapRectLayer = null;
}

function clearHeatmapAll() {
  heatmapData.value = null;
  heatmapRectLayer?.clearLayers();
}

function heatmapCellValue(row: number, col: number): number | null {
  const h = heatmapData.value;
  if (!h) return null;
  const g =
    heatmapScoreMode.value === "rule"
      ? h.business_score
      : heatmapScoreMode.value === "ml"
        ? h.business_score_ml
        : heatmapScoreMode.value === "stars"
          ? h.predicted_stars
          : h.survival_probability;
  if (!g) return null;
  const v = g[row]?.[col];
  return v == null || Number.isNaN(v) ? null : v;
}

function scoreToHeatColor(v: number | null): string | null {
  if (v == null) return null;
  let t: number;
  if (heatmapScoreMode.value === "survival") {
    t = Math.max(0, Math.min(1, v));
  } else if (heatmapScoreMode.value === "stars") {
    t = Math.max(0, Math.min(1, v / 5));
  } else {
    t = Math.max(0, Math.min(1, v / 100));
  }
  const h = (1 - t) * 240;
  return `hsl(${h}, 78%, 46%)`;
}

function drawHeatmapLayer() {
  if (!map || !heatmapRectLayer || !heatmapData.value) return;
  heatmapRectLayer.clearLayers();
  const h = heatmapData.value;
  const n = h.grid_size;
  const latStep = (h.max_lat - h.min_lat) / n;
  const lonStep = (h.max_lon - h.min_lon) / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = heatmapCellValue(i, j);
      const fill = scoreToHeatColor(v);
      if (fill == null) continue;
      const south = h.min_lat + i * latStep;
      const north = h.min_lat + (i + 1) * latStep;
      const west = h.min_lon + j * lonStep;
      const east = h.min_lon + (j + 1) * lonStep;
      L.rectangle(
        [
          [south, west],
          [north, east],
        ],
        { stroke: false, fillColor: fill, fillOpacity: 0.52, interactive: false }
      ).addTo(heatmapRectLayer);
    }
  }
}

async function runHeatmap() {
  const text = businessTypeText.value.trim();
  if (!text) {
    err.value = "Describe the business type first (same fields as pin mode).";
    return;
  }
  heatmapLoading.value = true;
  err.value = null;
  try {
    const req: MerchantHeatmapRequest = {
      city: city.value.trim() || null,
      state: stateParam(),
      category_query: text,
      category_keys: [],
      max_rows_if_no_city: maxRows.value,
      grid_size: heatmapGridSize.value,
    };
    if (priceTier.value) {
      req.price_level = parseInt(priceTier.value, 10);
    } else {
      const ppp = parseFloat(pricePerPerson.value.replace(",", "."));
      if (Number.isFinite(ppp) && ppp > 0) {
        req.price_per_person = ppp;
      }
    }
    heatmapData.value = await postMerchantHeatmap(req);
    await nextTick();
    drawHeatmapLayer();
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    heatmapLoading.value = false;
  }
}

function updatePin() {
  if (!map || !pinLayer) return;
  pinLayer.clearLayers();
  L.circleMarker([lat.value, lon.value], {
    radius: 9,
    color: "#b91c1c",
    weight: 2,
    fillColor: "#ef4444",
    fillOpacity: 0.95,
  }).addTo(pinLayer);
}

function stateParam(): string | null {
  const s = stateFilter.value.trim().toUpperCase();
  return s.length ? s : null;
}

async function loadCoverageLayers() {
  if (!map) return;
  coverageLoading.value = true;
  err.value = null;
  try {
    const cov = await getMerchantCoverage({
      city: city.value.trim() || null,
      state: stateParam(),
      max_rows_if_no_city: maxRows.value,
      max_sample_points: 450,
    });
    coverage.value = cov;

    if (hullLayer) {
      map.removeLayer(hullLayer);
      hullLayer = null;
    }
    if (sampleLayer) {
      map.removeLayer(sampleLayer);
      sampleLayer = null;
    }

    if (cov.hull_geojson) {
      hullLayer = L.geoJSON(cov.hull_geojson as unknown as GeoJsonObject, {
        style: {
          color: "#b91c1c",
          weight: 2,
          fillColor: "#fecaca",
          fillOpacity: 0.12,
        },
      }).addTo(map);
    }
    if (cov.sample_points_geojson) {
      sampleLayer = L.geoJSON(cov.sample_points_geojson as unknown as GeoJsonObject, {
        pointToLayer(_feat, ll) {
          return L.circleMarker(ll, {
            radius: 2,
            weight: 0,
            fillColor: "#78716c",
            fillOpacity: 0.45,
            color: "#78716c",
          });
        },
      }).addTo(map);
    }

    if (
      cov.geo_count > 0 &&
      cov.max_lat > cov.min_lat &&
      cov.max_lon > cov.min_lon &&
      Number.isFinite(cov.min_lat)
    ) {
      map.fitBounds(
        [
          [cov.min_lat, cov.min_lon],
          [cov.max_lat, cov.max_lon],
        ],
        { padding: [36, 36], maxZoom: 13 }
      );
    }
    updatePin();
    if (heatmapData.value) {
      void nextTick(() => drawHeatmapLayer());
    }
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    coverageLoading.value = false;
  }
}

async function initMap() {
  await nextTick();
  if (!mapEl.value) return;
  destroyMap();
  map = L.map(mapEl.value, { preferCanvas: true }).setView([lat.value, lon.value], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  map.createPane("merchantHeatmap");
  const heatPane = map.getPane("merchantHeatmap");
  if (heatPane) {
    heatPane.style.zIndex = "420";
    heatPane.style.pointerEvents = "none";
  }
  map.createPane("merchantPin");
  const pinPane = map.getPane("merchantPin");
  if (pinPane) pinPane.style.zIndex = "620";
  heatmapRectLayer = L.layerGroup([], { pane: "merchantHeatmap" }).addTo(map);
  pinLayer = L.layerGroup([], { pane: "merchantPin" }).addTo(map);
  map.on("click", (e: L.LeafletMouseEvent) => {
    lat.value = e.latlng.lat;
    lon.value = e.latlng.lng;
    updatePin();
    if (viewTab.value === "pin") {
      debouncedPredictFromMap();
    }
  });
  await loadCoverageLayers();
}

function debouncedPredictFromMap() {
  if (!businessTypeText.value.trim()) {
    err.value = "Enter a business type in the form (e.g. fast food, coffee), then click the map.";
    return;
  }
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    void run();
  }, 380);
}

async function run() {
  err.value = null;
  const text = businessTypeText.value.trim();
  if (!text) {
    err.value = "Describe the business you plan to open (e.g. fast food, pizza, coffee).";
    return;
  }
  loading.value = true;
  try {
    const req: Parameters<typeof postMerchantPredict>[0] = {
      city: city.value.trim() || null,
      state: stateParam(),
      lat: lat.value,
      lon: lon.value,
      category_query: text,
      category_keys: [],
      max_rows_if_no_city: maxRows.value,
    };
    if (priceTier.value) {
      req.price_level = parseInt(priceTier.value, 10);
    } else {
      const ppp = parseFloat(pricePerPerson.value.replace(",", "."));
      if (Number.isFinite(ppp) && ppp > 0) {
        req.price_per_person = ppp;
      }
    }
    result.value = await postMerchantPredict(req);
    resultDockOpen.value = true;
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

function closeResultDock() {
  resultDockOpen.value = false;
}

async function refreshCoverage() {
  if (map) {
    await loadCoverageLayers();
  } else {
    await initMap();
  }
}

function setRouteForCity(row: MerchantCityRow) {
  const q: Record<string, string> = { city: row.city };
  if (row.state) q.state = row.state;
  void router.replace({ path: route.path, query: q });
}

function applyCityRow(row: MerchantCityRow) {
  city.value = row.city;
  stateFilter.value = row.state || "";
  lat.value = row.center_lat;
  lon.value = row.center_lon;
  updatePin();
  void refreshCoverage();
  setRouteForCity(row);
}

function applyFromRoute() {
  const qc = route.query.city;
  const qs = route.query.state;
  const cname = typeof qc === "string" ? qc : Array.isArray(qc) ? (qc[0] ?? "") : "";
  const st = typeof qs === "string" ? qs : Array.isArray(qs) ? (qs[0] ?? "") : "";
  if (!cname || !cities.value.length) return;
  const stu = st.trim().toUpperCase();
  const row =
    cities.value.find((x) => x.city === cname && (!stu || (x.state || "").toUpperCase() === stu)) ??
    cities.value.find((x) => x.city.toLowerCase() === cname.toLowerCase());
  if (!row) return;
  city.value = row.city;
  stateFilter.value = row.state || "";
  lat.value = row.center_lat;
  lon.value = row.center_lon;
  if (map) {
    updatePin();
    void loadCoverageLayers();
  }
}

function letterDomId(letter: string): string {
  return letter === "#" ? "merchant-letter-sym" : `merchant-letter-${letter}`;
}

function scrollToLetter(letter: string) {
  document.getElementById(letterDomId(letter))?.scrollIntoView({ behavior: "smooth", block: "start" });
}

watch([city, maxRows, stateFilter], () => {
  clearHeatmapAll();
  if (map) void refreshCoverage();
});

watch(heatmapScoreMode, () => {
  if (heatmapData.value) drawHeatmapLayer();
});

watch(viewTab, (t) => {
  if (t === "heatmap" && heatmapData.value) {
    void nextTick(() => drawHeatmapLayer());
  }
});

watch(
  () => route.query,
  () => {
    applyFromRoute();
  },
  { deep: true }
);

watch(result, () => {
  if (result.value) resultDockOpen.value = true;
});

onMounted(async () => {
  try {
    const res = await getMerchantCities({ min_rows: SPATIAL_CITY_MIN_TRAIN_ROWS });
    cities.value = res.cities;
  } catch {
    cities.value = [];
  }
  applyFromRoute();
  await initMap();
});

onBeforeUnmount(() => {
  destroyMap();
  if (debounceTimer) clearTimeout(debounceTimer);
});
</script>

<template>
  <div class="page">
    <nav class="back">
      <router-link to="/">← Home</router-link>
    </nav>

    <header class="head">
      <h1>Merchant site predictor</h1>
      <p class="muted">
        <strong>1)</strong> Enter a <strong>restaurant or venue type</strong> in plain language.
        <strong>2)</strong> Choose an area in the city list (updates the map slice).
        <strong>3)</strong> Use <strong>Pin &amp; detail</strong> to click the map for a single site, or
        <strong>Region heatmap</strong> for a macro view over the same slice. Red outline: training hull; gray
        dots: sample points.
      </p>
    </header>

    <div class="view-tabs" role="tablist" aria-label="Predictor mode">
      <button
        type="button"
        role="tab"
        class="tab-btn"
        :class="{ active: viewTab === 'pin' }"
        :aria-selected="viewTab === 'pin'"
        @click="viewTab = 'pin'"
      >
        Pin &amp; detail
      </button>
      <button
        type="button"
        role="tab"
        class="tab-btn"
        :class="{ active: viewTab === 'heatmap' }"
        :aria-selected="viewTab === 'heatmap'"
        @click="viewTab = 'heatmap'"
      >
        Region heatmap
      </button>
    </div>

    <div class="layout">
      <section class="map-section">
        <div class="map-toolbar">
          <span v-if="coverageLoading" class="pill">Loading coverage…</span>
          <span v-else-if="coverage" class="pill muted-pill">
            {{ coverage.reference_count }} ref. rows · {{ coverage.geo_count }} with coords
            <template v-if="coverage.valid_hull"> · hull OK</template>
          </span>
          <span v-if="viewTab === 'heatmap' && heatmapData" class="pill muted-pill">
            {{ heatmapData.grid_size }}×{{ heatmapData.grid_size }} grid
            <template v-if="heatmapResolvedLine"> · {{ heatmapResolvedLine }}</template>
          </span>
          <span v-if="result" class="pill muted-pill result-inline-pill">
            <template v-if="result.business_score != null">Rule {{ result.business_score.toFixed(0) }}</template
            ><template v-if="result.business_score != null && result.business_score_ml != null"> · </template
            ><template v-if="result.business_score_ml != null">ML {{ result.business_score_ml.toFixed(0) }}</template
            ><template v-if="result.business_score != null || result.business_score_ml != null"> · </template
            >{{ (result.survival_probability * 100).toFixed(1) }}% · ★ {{ result.predicted_stars.toFixed(2) }}
            <span v-if="resolvedCategoryLine" class="inline-muted">· {{ resolvedCategoryLine }}</span>
          </span>
          <button type="button" class="btn-ghost" :disabled="coverageLoading" @click="refreshCoverage">
            Reload map
          </button>
        </div>

        <p v-if="viewTab === 'pin'" class="map-lead">
          Click the map to set the site (prediction runs a moment after the click if a type is filled in).
        </p>
        <p v-else class="map-lead heatmap-lead">
          Use the same <strong>type and price</strong> in the panel on the right, then <strong>Generate heatmap</strong>.
          Warmer = higher for the selected metric. Hull-only cells: empty regions are outside the training hull.
        </p>

        <div v-if="viewTab === 'heatmap'" class="heatmap-toolbar">
          <div class="heatmap-toolbar-row">
            <label class="heatmap-label" for="hm-metric">Color by</label>
            <select id="hm-metric" v-model="heatmapScoreMode" class="inp heatmap-select" aria-label="Heatmap metric">
              <option value="rule">Rule score (0–100)</option>
              <option value="ml">ML score (0–100)</option>
              <option value="survival">Survival probability</option>
              <option value="stars">Predicted stars (0–5)</option>
            </select>
            <label class="heatmap-label" for="hm-grid">Grid</label>
            <select id="hm-grid" v-model.number="heatmapGridSize" class="inp heatmap-select" aria-label="Grid size">
              <option :value="8">8×8 (faster)</option>
              <option :value="12">12×12</option>
              <option :value="16">16×16 (slower)</option>
            </select>
            <button
              type="button"
              class="btn-heatmap"
              :disabled="heatmapLoading || coverageLoading"
              @click="void runHeatmap()"
            >
              {{ heatmapLoading ? "Computing…" : "Generate heatmap" }}
            </button>
            <button
              v-if="heatmapData"
              type="button"
              class="btn-ghost"
              :disabled="heatmapLoading"
              @click="clearHeatmapAll"
            >
              Clear overlay
            </button>
          </div>
        </div>

        <div class="map-wrap">
          <div
            ref="mapEl"
            class="map-host"
            role="application"
            :aria-label="viewTab === 'pin' ? 'Map: click to set site' : 'Map: region heatmap'"
          />
          <div
            v-if="viewTab === 'heatmap' && heatmapData"
            class="heatmap-legend"
            role="img"
            :aria-label="heatmapMetricTitle + ', lower to higher'"
          >
            <div class="heatmap-legend-title">{{ heatmapMetricTitle }}</div>
            <div class="heatmap-legend-bar" aria-hidden="true" />
            <div class="heatmap-legend-ticks">
              <span>{{ heatmapLegendEnds.lo }}</span>
              <span>{{ heatmapLegendEnds.hi }}</span>
            </div>
          </div>
          <aside
            v-if="result && resultDockOpen && viewTab === 'pin'"
            class="result-dock"
            aria-live="polite"
            aria-label="Prediction result"
          >
            <div class="result-dock-head">
              <span class="result-dock-title">Prediction</span>
              <button type="button" class="dock-close" aria-label="Close result panel" @click="closeResultDock">×</button>
            </div>
            <div class="stats stats--dock">
              <p v-if="!result.inside_reference_hull" class="warn" role="status">
                Pin is <strong>outside</strong> the training hull — treat opening/survival probability as exploratory
                only.
              </p>
              <div class="stat">
                <div class="stat-label">Opening / survival P(still open)</div>
                <div class="stat-value">{{ (result.survival_probability * 100).toFixed(1) }}%</div>
                <p class="stat-note">Same score as the survival (binary) head.</p>
              </div>
              <div class="stat">
                <div class="stat-label">Predicted stars</div>
                <div class="stat-value">{{ result.predicted_stars.toFixed(2) }} / 5</div>
              </div>
              <div v-if="result.business_score != null" class="stat">
                <div class="stat-label">Composite score — rules (0–100)</div>
                <div class="stat-value">{{ result.business_score.toFixed(0) }}</div>
                <p class="stat-note">Rule-based blend of survival, stars, price vs neighborhood, and risk.</p>
              </div>
              <div v-if="result.business_score_ml != null" class="stat">
                <div class="stat-label">ML score (0–100)</div>
                <div class="stat-value">{{ result.business_score_ml.toFixed(0) }}</div>
                <p class="stat-note">Supervised P(still open) from location and category features only (not using survival/rating model outputs as inputs).</p>
              </div>
              <div
                v-if="
                  result.nearby_avg_price_level != null || result.price_fit != null || result.price_gap != null
                "
                class="stat"
              >
                <div class="stat-label">Price and neighborhood</div>
                <p class="stat-note" style="margin: 0 0 0.5rem">
                  <template v-if="result.nearby_avg_price_level != null"
                    >1 km neighborhood average price tier is about <strong>{{ result.nearby_avg_price_level.toFixed(1) }}</strong> (1=lowest)</template
                  >
                  <template
                    v-if="result.price_gap != null"
                  >; compared to your tier, about <strong>{{ result.price_gap >= 0 ? "+" : "" }}{{ result.price_gap.toFixed(1) }}</strong> tier(s) off</template
                  >
                  <template v-if="result.price_fit"
                    >; local match: <strong>{{ priceFitLabel(result.price_fit) }}</strong></template
                  >
                </p>
              </div>
              <ul v-if="riskEntries.length" class="risk-list" role="list">
                <li v-for="[k, v] in riskEntries" :key="k" class="risk-item">
                  <span class="risk-k">{{ k }}</span>
                  <span class="risk-v">{{ v }}</span>
                </li>
              </ul>
              <p v-if="result.explanation" class="explanation-box">{{ result.explanation }}</p>
            </div>
          </aside>
        </div>
        <p class="map-hint">
          Prefer pins <strong>inside</strong> the red hull — outside, scores are extrapolated and less reliable.
        </p>
      </section>

      <aside class="side">
        <div class="panel-cities">
          <div class="panel-head">
            <h2 class="panel-title">Cities</h2>
            <span class="panel-meta">{{ cities.length }} areas</span>
          </div>
          <input
            v-model="citySearch"
            class="inp city-search"
            type="search"
            placeholder="Filter cities…"
            aria-label="Filter city list"
          />
          <div v-if="citiesByLetter.letters.length" class="letter-bar" role="navigation" aria-label="Jump by letter">
            <button
              v-for="L in citiesByLetter.letters"
              :key="L"
              type="button"
              class="letter-btn"
              @click="scrollToLetter(L)"
            >
              {{ L }}
            </button>
          </div>
          <div class="city-scroll">
            <template v-for="L in citiesByLetter.letters" :key="'g-' + L">
              <div :id="letterDomId(L)" class="letter-block">
                <h3 class="letter-h">{{ L }}</h3>
                <div class="city-chips">
                  <button
                    v-for="c in citiesByLetter.groups.get(L) || []"
                    :key="c.city + '|' + (c.state || '')"
                    type="button"
                    class="city-chip"
                    :class="{ active: c.city === city && (c.state || '') === (stateFilter || '') }"
                    @click="applyCityRow(c)"
                  >
                    <span class="chip-name">{{ c.city }}</span>
                    <span v-if="c.state" class="chip-st">{{ c.state }}</span>
                    <span class="chip-n">{{ c.row_count }}</span>
                  </button>
                </div>
              </div>
            </template>
            <p v-if="!filteredCities.length" class="empty-cities">No cities match this filter.</p>
          </div>
        </div>

        <div class="form-card form-card--minimal">
          <p class="form-step">Restaurant type (only text field)</p>
          <p class="form-hint">
            The server matches this to Yelp-style training columns for the selected city (e.g. <strong>burger</strong>,
            <strong>fast food</strong>, <strong>coffee</strong>). Use commas for several types. Then
            <strong>click the map</strong> — no Run button.
          </p>
          <input
            v-model="businessTypeText"
            class="inp"
            type="text"
            autocomplete="off"
            placeholder="e.g. fast food, coffee"
            aria-label="Restaurant or venue type"
            @keydown.enter.prevent="void run()"
          />
          <p v-if="resolvedCategoryLine" class="resolved-line">Last run used: {{ resolvedCategoryLine }}</p>
          <p class="form-step" style="margin-top: 1rem">Price (optional)</p>
          <p class="form-hint">Pick a price tier, or only fill approximate USD per person — one or the other. Used vs the 1 km neighborhood mean.</p>
          <div class="price-row">
            <label class="field-label" for="price-tier">Price tier 1–4</label>
            <select id="price-tier" v-model="priceTier" class="inp price-select" aria-label="Price tier 1 to 4">
              <option value="">None (use per-person price below)</option>
              <option value="1">1 — lowest</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4 — highest</option>
            </select>
          </div>
          <div v-if="!priceTier" class="price-row">
            <label class="field-label" for="price-ppp">Price per person (approx. USD)</label>
            <input
              id="price-ppp"
              v-model="pricePerPerson"
              class="inp"
              type="text"
              inputmode="decimal"
              placeholder="e.g. 15"
              aria-label="Approx price per person USD"
            />
          </div>
        </div>

        <p v-if="err" class="err">{{ err }}</p>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.page {
  margin: 0 auto;
  padding: 1.75rem 1.25rem 3rem;
  box-sizing: border-box;
  max-width: min(88rem, 100%);
  overflow-x: hidden;
}

.back {
  margin-bottom: 1rem;
}

.back a {
  color: #dc2626;
  font-weight: 600;
  font-size: 0.875rem;
  text-decoration: none;
}

.back a:hover {
  text-decoration: underline;
}

.head h1 {
  margin: 0 0 0.5rem;
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #1c1917;
}

.muted {
  color: #78716c;
  font-size: 0.9rem;
  line-height: 1.55;
  margin: 0;
}

.muted code {
  font-size: 0.82em;
  background: #f5f5f4;
  padding: 0.1em 0.3em;
  border-radius: 4px;
}

.layout {
  display: grid;
  /* Wider map column on desktop */
  grid-template-columns: minmax(0, 2.35fr) minmax(0, min(21rem, 100%));
  gap: 1.5rem;
  align-items: start;
  margin-top: 1.25rem;
  width: 100%;
  min-width: 0;
}

@media (max-width: 960px) {
  .layout {
    grid-template-columns: 1fr;
  }
}

.map-section {
  min-width: 0;
}

.map-toolbar {
  margin-bottom: 0.5rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.6rem;
  width: 100%;
}

.pill {
  font-size: 0.78rem;
  font-weight: 700;
  padding: 0.28rem 0.65rem;
  border-radius: 999px;
  background: #fef2f2;
  color: #991b1b;
}

.muted-pill {
  background: #f5f5f4;
  color: #57534e;
}

.map-host {
  height: clamp(22rem, 70vh, 52rem);
  min-height: 22rem;
  width: 100%;
  max-width: 100%;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid #c9c5c2;
  box-shadow:
    0 22px 64px rgba(28, 25, 23, 0.26),
    0 10px 28px rgba(28, 25, 23, 0.16),
    0 2px 6px rgba(28, 25, 23, 0.12),
    0 0 0 1px rgba(255, 255, 255, 0.55) inset;
  z-index: 0;
}

.map-wrap {
  position: relative;
  width: 100%;
}

.result-dock {
  position: absolute;
  top: 10px;
  right: 10px;
  width: min(18.5rem, calc(100% - 20px));
  max-height: min(58vh, calc(100% - 20px));
  overflow-x: hidden;
  overflow-y: auto;
  padding: 0.65rem 0.7rem 0.75rem;
  box-sizing: border-box;
  background: rgba(255, 255, 255, 0.97);
  border: 1px solid #e7e5e4;
  border-radius: 16px;
  box-shadow:
    0 16px 48px rgba(28, 25, 23, 0.2),
    0 6px 16px rgba(28, 25, 23, 0.12);
  z-index: 1100;
  pointer-events: auto;
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
}

.result-dock-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}

.dock-close {
  flex-shrink: 0;
  width: 1.75rem;
  height: 1.75rem;
  border: none;
  border-radius: 8px;
  background: #f5f5f4;
  color: #57534e;
  font-size: 1.1rem;
  line-height: 1;
  cursor: pointer;
}

.dock-close:hover {
  background: #e7e5e4;
  color: #1c1917;
}

.result-dock-title {
  font-size: 0.72rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #a8a29e;
}

.stats--dock .warn {
  margin-bottom: 0.15rem;
}

.subhead--dock {
  margin: 0.65rem 0 0.35rem;
  font-size: 0.82rem;
}

.code--dock {
  max-height: 10rem;
  margin: 0;
}

.map-lead {
  margin: 0.35rem 0 0.4rem;
  font-size: 0.8rem;
  line-height: 1.4;
  color: #57534e;
}

.heatmap-lead strong {
  color: #44403c;
}

.view-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin: 1rem 0 0.25rem;
  padding: 0.2rem;
  background: #f5f5f4;
  border-radius: 12px;
  border: 1px solid #e7e5e4;
  width: fit-content;
  max-width: 100%;
}

.tab-btn {
  padding: 0.45rem 0.9rem;
  font-size: 0.85rem;
  font-weight: 700;
  border: none;
  border-radius: 9px;
  background: transparent;
  color: #78716c;
  cursor: pointer;
}

.tab-btn:hover {
  color: #44403c;
  background: rgba(255, 255, 255, 0.65);
}

.tab-btn.active {
  background: #fff;
  color: #b91c1c;
  box-shadow: 0 1px 3px rgba(28, 25, 23, 0.12);
}

.heatmap-toolbar {
  margin: 0.5rem 0 0.65rem;
  padding: 0.65rem 0.75rem;
  background: #fafaf9;
  border: 1px solid #e7e5e4;
  border-radius: 12px;
}

.heatmap-toolbar-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.65rem;
}

.heatmap-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #57534e;
}

.heatmap-select {
  width: auto;
  min-width: 9.5rem;
  font-size: 0.8rem;
  padding: 0.35rem 0.5rem;
}

.btn-heatmap {
  padding: 0.4rem 0.85rem;
  font-size: 0.8rem;
  font-weight: 700;
  border: none;
  border-radius: 8px;
  background: #b91c1c;
  color: #fff;
  cursor: pointer;
}

.btn-heatmap:hover:not(:disabled) {
  background: #991b1b;
}

.btn-heatmap:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.heatmap-legend {
  position: absolute;
  z-index: 1150;
  left: 10px;
  bottom: 10px;
  padding: 0.5rem 0.55rem 0.45rem;
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid #e7e5e4;
  border-radius: 10px;
  box-shadow: 0 6px 20px rgba(28, 25, 23, 0.12);
  min-width: 9.5rem;
  pointer-events: none;
}

.heatmap-legend-title {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #57534e;
  margin-bottom: 0.35rem;
}

.heatmap-legend-bar {
  height: 10px;
  border-radius: 5px;
  background: linear-gradient(90deg, hsl(240, 78%, 46%), hsl(120, 78%, 44%), hsl(0, 78%, 48%));
}

.heatmap-legend-ticks {
  display: flex;
  justify-content: space-between;
  margin-top: 0.2rem;
  font-size: 0.65rem;
  color: #78716c;
}

.btn-ghost {
  margin-left: auto;
  padding: 0.35rem 0.6rem;
  font-size: 0.75rem;
  font-weight: 600;
  border: 1px solid #d6d3d1;
  border-radius: 8px;
  background: #fff;
  color: #44403c;
  cursor: pointer;
}

.btn-ghost:hover:not(:disabled) {
  border-color: #fca5a5;
  color: #b91c1c;
}

.btn-ghost:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.result-inline-pill {
  max-width: min(22rem, 100%);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-inline-pill .inline-muted {
  font-weight: 600;
  color: #78716c;
}

.map-hint {
  margin: 0.55rem 0 0;
  font-size: 0.78rem;
  color: #78716c;
  line-height: 1.45;
}

.side {
  min-width: 0;
  max-width: 100%;
  overflow-x: hidden;
}

.panel-cities {
  margin-bottom: 1rem;
  padding: 0.9rem 0.85rem 0.75rem;
  background: #fff;
  border: 1px solid #e7e5e4;
  border-radius: 16px;
  box-shadow:
    0 16px 48px rgba(28, 25, 23, 0.2),
    0 6px 16px rgba(28, 25, 23, 0.1),
    0 1px 3px rgba(28, 25, 23, 0.08);
  min-width: 0;
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.panel-title {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 800;
  color: #1c1917;
}

.panel-meta {
  font-size: 0.72rem;
  font-weight: 600;
  color: #a8a29e;
}

.city-search {
  margin-bottom: 0.45rem;
}

.letter-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-bottom: 0.45rem;
  max-height: 4.5rem;
  overflow-y: auto;
}

.letter-btn {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 0.2rem 0.45rem;
  border-radius: 6px;
  border: 1px solid #e7e5e4;
  background: #fafaf9;
  color: #44403c;
  cursor: pointer;
}

.letter-btn:hover {
  border-color: #fca5a5;
  color: #b91c1c;
}

.city-scroll {
  max-height: 14rem;
  overflow-y: auto;
  overflow-x: hidden;
  min-width: 0;
  padding: 0.25rem 0.15rem 0.35rem;
  border-radius: 10px;
  background: #fafaf9;
}

.letter-block {
  scroll-margin-top: 0.35rem;
}

.letter-h {
  margin: 0.5rem 0 0.25rem;
  font-size: 0.72rem;
  font-weight: 800;
  color: #a8a29e;
  letter-spacing: 0.06em;
}

.city-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.city-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  max-width: 100%;
  padding: 0.28rem 0.45rem;
  border-radius: 999px;
  border: 1px solid #e7e5e4;
  background: #fff;
  font-size: 0.72rem;
  cursor: pointer;
  color: #44403c;
  min-width: 0;
}

.city-chip:hover {
  border-color: #fca5a5;
}

.city-chip.active {
  border-color: #dc2626;
  background: #fef2f2;
  color: #991b1b;
}

.chip-name {
  font-weight: 700;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 9rem;
}

.chip-st {
  font-weight: 600;
  color: #78716c;
}

.chip-n {
  font-size: 0.65rem;
  color: #a8a29e;
}

.empty-cities {
  margin: 0.5rem 0 0;
  font-size: 0.78rem;
  color: #a8a29e;
}

.form-card {
  padding: 1.2rem 1.1rem;
  background: #fff;
  border: 1px solid #e7e5e4;
  border-radius: 18px;
  box-shadow:
    0 18px 52px rgba(28, 25, 23, 0.2),
    0 8px 22px rgba(28, 25, 23, 0.12),
    0 2px 6px rgba(28, 25, 23, 0.08);
  display: grid;
  gap: 0.5rem;
  min-width: 0;
  max-width: 100%;
  overflow-x: hidden;
}

.form-card--minimal {
  padding: 0.95rem 1rem 1.05rem;
}

.form-step {
  margin: 0 0 0.15rem;
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #78716c;
}

.form-step + .form-hint {
  margin-top: 0;
}

.form-hint {
  margin: 0 0 0.45rem;
  font-size: 0.75rem;
  line-height: 1.45;
  color: #a8a29e;
}

.form-step:not(:first-child) {
  margin-top: 0.5rem;
}

.resolved-line {
  margin: 0.1rem 0 0.35rem;
  font-size: 0.75rem;
  line-height: 1.35;
  color: #57534e;
  font-style: italic;
}

.form-card label {
  font-weight: 700;
  font-size: 0.8rem;
  color: #44403c;
  margin-top: 0.35rem;
}

.form-card label:first-of-type {
  margin-top: 0;
}

.inp {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  padding: 0.55rem 0.65rem;
  border: 1px solid #d6d3d1;
  border-radius: 10px;
  font-family: inherit;
  font-size: 0.9rem;
}

.inp:focus {
  outline: 2px solid rgba(220, 38, 38, 0.35);
  border-color: #fca5a5;
}

.state-inp {
  max-width: 6rem;
  text-transform: uppercase;
}

.row {
  display: flex;
  gap: 0.65rem;
  min-width: 0;
}

.row .inp {
  flex: 1 1 0;
}

.inp-coord {
  font-size: 0.78rem;
}

.short {
  max-width: 100%;
}

.submit {
  margin-top: 0.65rem;
  padding: 0.65rem 1.1rem;
  border: none;
  border-radius: 999px;
  background: linear-gradient(135deg, #b91c1c, #ef4444);
  color: #fff;
  font-weight: 700;
  font-size: 0.9rem;
  cursor: pointer;
  box-shadow:
    0 14px 36px rgba(127, 29, 29, 0.45),
    0 6px 18px rgba(185, 28, 28, 0.38),
    0 2px 6px rgba(220, 38, 38, 0.3);
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}

.submit:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.btn-secondary {
  margin-top: 0.35rem;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  border: 1px solid #e7e5e4;
  background: #fafaf9;
  font-weight: 600;
  font-size: 0.82rem;
  cursor: pointer;
  color: #44403c;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  box-shadow:
    0 8px 22px rgba(28, 25, 23, 0.12),
    0 2px 6px rgba(28, 25, 23, 0.08);
}

.btn-secondary:hover:not(:disabled) {
  box-shadow:
    0 12px 30px rgba(28, 25, 23, 0.16),
    0 4px 10px rgba(28, 25, 23, 0.1);
}

.btn-secondary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.err {
  color: #b91c1c;
  margin-top: 0.85rem;
  font-size: 0.88rem;
  word-break: break-word;
}

.stats {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-top: 1rem;
  min-width: 0;
}

.stats.stats--dock {
  margin-top: 0;
  gap: 0.55rem;
}

.warn {
  margin: 0;
  padding: 0.65rem 0.75rem;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 12px;
  font-size: 0.82rem;
  color: #92400e;
  line-height: 1.45;
  overflow-wrap: anywhere;
  box-shadow:
    0 10px 32px rgba(120, 53, 15, 0.22),
    0 3px 10px rgba(146, 64, 14, 0.14);
}

.stat {
  background: #fff;
  border: 1px solid #e7e5e4;
  border-radius: 16px;
  padding: 1rem 0.95rem;
  min-width: 0;
  box-shadow:
    0 14px 40px rgba(28, 25, 23, 0.18),
    0 6px 16px rgba(28, 25, 23, 0.1),
    0 1px 4px rgba(28, 25, 23, 0.07);
}

.stat-small .stat-value {
  font-size: 1.1rem;
}

.stat-label {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #a8a29e;
}

.stat-value {
  font-size: 1.35rem;
  font-weight: 800;
  margin-top: 0.25rem;
  color: #1c1917;
  word-break: break-word;
}

.stat-note {
  margin: 0.35rem 0 0;
  font-size: 0.72rem;
  color: #a8a29e;
}

.subhead {
  margin: 1.25rem 0 0.5rem;
  font-size: 0.95rem;
  font-weight: 800;
  color: #44403c;
}

.code {
  background: #1c1917;
  color: #e7e5e4;
  padding: 0.85rem;
  border-radius: 14px;
  font-size: 0.72rem;
  overflow: auto;
  line-height: 1.45;
  max-height: 14rem;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  box-shadow:
    0 18px 48px rgba(0, 0, 0, 0.38),
    0 6px 16px rgba(28, 25, 23, 0.22);
}

.price-row {
  margin-top: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  max-width: 100%;
}

.field-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: #57534e;
}

.price-select {
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}

.risk-list {
  margin: 0;
  padding: 0.55rem 0.65rem;
  list-style: none;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 12px;
  font-size: 0.78rem;
  line-height: 1.45;
}

.risk-item {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.5rem;
  margin: 0.2rem 0;
}

.risk-k {
  font-weight: 700;
  color: #92400e;
}

.risk-v {
  color: #44403c;
  flex: 1;
  min-width: 8rem;
}

.explanation-box {
  margin: 0.35rem 0 0;
  padding: 0.65rem 0.75rem;
  background: #f5f5f4;
  border: 1px solid #e7e5e4;
  border-radius: 12px;
  font-size: 0.8rem;
  line-height: 1.5;
  color: #44403c;
  overflow-wrap: anywhere;
}
</style>
