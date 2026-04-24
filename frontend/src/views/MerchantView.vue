<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import type { GeoJsonObject } from "geojson";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  getMerchantCities,
  getMerchantCoverage,
  postMerchantPredict,
  type MerchantCityRow,
  type MerchantCoverageResponse,
  type MerchantPredictResponse,
} from "../api/client";

const route = useRoute();
const router = useRouter();

const city = ref("Philadelphia");
const stateFilter = ref("");
const lat = ref(39.9526);
const lon = ref(-75.1652);
/** Comma or newline separated cat_* column names */
const categoriesText = ref("cat_coffee_&_tea, cat_fast_food");
const maxRows = ref(2000);
const loading = ref(false);
const coverageLoading = ref(false);
const err = ref<string | null>(null);
const result = ref<MerchantPredictResponse | null>(null);
const coverage = ref<MerchantCoverageResponse | null>(null);

const cities = ref<MerchantCityRow[]>([]);
const citySearch = ref("");

/** Toolbar summary popover: closed by user until the next prediction. */
const teaserPopoverDismissed = ref(false);
const teaserMouseOver = ref(false);
const teaserFocusInside = ref(false);

const teaserPopoverVisible = computed(
  () => !!result.value && !teaserPopoverDismissed.value && (teaserMouseOver.value || teaserFocusInside.value)
);

function onTeaserMouseEnter() {
  teaserMouseOver.value = true;
}

function onTeaserMouseLeave(e: MouseEvent) {
  const el = e.currentTarget as HTMLElement;
  if (e.relatedTarget instanceof Node && el.contains(e.relatedTarget)) return;
  teaserMouseOver.value = false;
}

function onTeaserFocusIn() {
  teaserFocusInside.value = true;
}

function onTeaserFocusOut(e: FocusEvent) {
  const el = e.currentTarget as HTMLElement;
  if (e.relatedTarget instanceof Node && el.contains(e.relatedTarget)) return;
  teaserFocusInside.value = false;
}

function dismissTeaserPopover() {
  teaserPopoverDismissed.value = true;
  teaserMouseOver.value = false;
  teaserFocusInside.value = false;
}

const mapEl = ref<HTMLElement | null>(null);
let map: L.Map | null = null;
let hullLayer: L.Layer | null = null;
let sampleLayer: L.Layer | null = null;
let pinLayer: L.LayerGroup | null = null;

const categoryKeys = computed(() =>
  categoriesText.value
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
);

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
  pinLayer = L.layerGroup().addTo(map);
  map.on("click", (e: L.LeafletMouseEvent) => {
    lat.value = e.latlng.lat;
    lon.value = e.latlng.lng;
    updatePin();
    debouncedPredictFromMap();
  });
  await loadCoverageLayers();
}

function debouncedPredictFromMap() {
  if (!categoryKeys.value.length) {
    err.value = "Add at least one category column before clicking the map.";
    return;
  }
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    void run();
  }, 380);
}

async function run() {
  err.value = null;
  const keys = categoryKeys.value;
  if (!keys.length) {
    err.value = "Enter at least one category column (e.g. cat_fast_food).";
    return;
  }
  loading.value = true;
  try {
    result.value = await postMerchantPredict({
      city: city.value.trim() || null,
      state: stateParam(),
      lat: lat.value,
      lon: lon.value,
      category_keys: keys,
      max_rows_if_no_city: maxRows.value,
    });
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
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
  if (map) void refreshCoverage();
});

watch(
  () => route.query,
  () => {
    applyFromRoute();
  },
  { deep: true }
);

watch(
  result,
  () => {
    teaserPopoverDismissed.value = false;
  },
  { deep: true }
);

onMounted(async () => {
  try {
    const res = await getMerchantCities({ min_rows: 10 });
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
        Click the map to set a pin — same
        <code>POST /api/v1/merchant/predict</code>. Red outline: convex hull of training coordinates for the
        current city slice. Gray dots: sampled training locations. Use the city list to jump; URL updates as
        <code>?city=…&amp;state=…</code> for sharing.
      </p>
    </header>

    <div class="layout">
      <section class="map-section">
        <div class="map-toolbar">
          <span v-if="coverageLoading" class="pill">Loading coverage…</span>
          <span v-else-if="coverage" class="pill muted-pill">
            {{ coverage.reference_count }} ref. rows · {{ coverage.geo_count }} with coords
            <template v-if="coverage.valid_hull"> · hull OK</template>
          </span>
          <div
            v-if="result"
            class="result-teaser-wrap"
            @mouseenter="onTeaserMouseEnter"
            @mouseleave="onTeaserMouseLeave"
            @focusin="onTeaserFocusIn"
            @focusout="onTeaserFocusOut"
          >
            <span class="result-teaser pill muted-pill" tabindex="0">
              Latest: {{ (result.survival_probability * 100).toFixed(1) }}% · ★ {{ result.predicted_stars.toFixed(2) }}
            </span>
            <div v-show="teaserPopoverVisible" class="result-popover" role="tooltip">
              <button
                type="button"
                class="popover-close"
                aria-label="Close summary popover"
                @click.stop="dismissTeaserPopover"
              >
                ×
              </button>
              <div class="popover-stats">
                <div><strong>Survival</strong> {{ (result.survival_probability * 100).toFixed(1) }}%</div>
                <div><strong>Stars</strong> {{ result.predicted_stars.toFixed(2) }} / 5</div>
                <div><strong>Ref. rows</strong> {{ result.reference_row_count }}</div>
                <div><strong>In hull</strong> {{ result.inside_reference_hull ? "Yes" : "No" }}</div>
              </div>
            </div>
          </div>
        </div>
        <div class="map-wrap">
          <div ref="mapEl" class="map-host" role="application" aria-label="Map: click to select location" />
          <aside v-if="result" class="result-dock" aria-live="polite" aria-label="Prediction result">
            <div class="result-dock-head">
              <span class="result-dock-title">Prediction</span>
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
              <div class="stat">
                <div class="stat-label">Reference rows</div>
                <div class="stat-value">{{ result.reference_row_count }}</div>
              </div>
              <div class="stat stat-small">
                <div class="stat-label">Inside data hull</div>
                <div class="stat-value">{{ result.inside_reference_hull ? "Yes" : "No" }}</div>
              </div>
            </div>
            <template v-if="Object.keys(result.live_feature_preview || {}).length">
              <h2 class="subhead subhead--dock">Feature preview</h2>
              <pre class="code code--dock">{{ JSON.stringify(result.live_feature_preview, null, 2) }}</pre>
            </template>
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

        <div class="form-card">
          <label>City (filter reference businesses, optional)</label>
          <input v-model="city" class="inp" type="text" placeholder="Philadelphia" />

          <label>State (optional, disambiguate same city name)</label>
          <input
            v-model="stateFilter"
            class="inp state-inp"
            type="text"
            maxlength="2"
            placeholder="PA"
            autocapitalize="characters"
          />

          <label>Latitude / longitude</label>
          <div class="row">
            <input v-model.number="lat" class="inp inp-coord" type="number" step="any" @input="updatePin" />
            <input v-model.number="lon" class="inp inp-coord" type="number" step="any" @input="updatePin" />
          </div>

          <label>Category columns (<code>cat_*</code>, comma or newline)</label>
          <textarea v-model="categoriesText" class="inp textarea-cats" rows="4" placeholder="cat_coffee_&_tea&#10;cat_fast_food" />

          <label>Max reference rows when no city column</label>
          <input v-model.number="maxRows" class="inp short" type="number" min="100" max="50000" />

          <button type="button" class="submit" :disabled="loading" @click="run">
            {{ loading ? "Running…" : "Run prediction" }}
          </button>
          <button type="button" class="btn-secondary" :disabled="coverageLoading" @click="refreshCoverage">
            Reload map coverage
          </button>
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
  gap: 0.5rem;
  align-items: center;
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
  margin-bottom: 0.45rem;
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

.result-teaser-wrap {
  position: relative;
}

.result-teaser {
  cursor: default;
}

.result-popover {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  z-index: 1200;
  min-width: 11.5rem;
  padding: 0.55rem 2.1rem 0.65rem 0.7rem;
  box-sizing: border-box;
  background: #fff;
  border: 1px solid #e7e5e4;
  border-radius: 12px;
  font-size: 0.78rem;
  line-height: 1.5;
  color: #44403c;
  box-shadow:
    0 14px 40px rgba(28, 25, 23, 0.18),
    0 4px 12px rgba(28, 25, 23, 0.1);
}

.popover-close {
  position: absolute;
  top: 0.3rem;
  right: 0.3rem;
  width: 1.65rem;
  height: 1.65rem;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: #78716c;
  font-size: 1.15rem;
  line-height: 1;
  cursor: pointer;
}

.popover-close:hover {
  background: #f5f5f4;
  color: #1c1917;
}

.popover-stats strong {
  color: #1c1917;
  font-weight: 700;
  margin-right: 0.25rem;
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

.textarea-cats {
  resize: vertical;
  overflow-wrap: anywhere;
  word-break: break-word;
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
</style>
