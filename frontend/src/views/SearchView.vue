<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  getStates,
  postSearch,
  type SearchActionEvent,
  type SearchRequest,
  type SearchResponse,
} from "../api/client";

const CUISINES = [
  "Sushi",
  "Steakhouse",
  "Korean",
  "Fast Food",
  "Chinese",
  "Burger",
  "Healthy",
] as const;

const states = ref<string[]>([]);
const browseState = ref("PA");
const browseCity = ref("");
const nlQuery = ref("");
const keywords = ref("");
const selectedCuisines = ref<string[]>([]);
const topK = ref(10);
const poolK = ref(45);
const wSemantic = ref(0.85);
const wRating = ref(1.05);
const wPrice = ref(0.15);
const wDistance = ref(0.2);
const wPopularity = ref(0.1);
const forceRebuild = ref(false);
const step2Open = ref(true);

const loading = ref(false);
const err = ref<string | null>(null);
const data = ref<SearchResponse | null>(null);
const lastMode = ref<"discover" | "refine" | null>(null);

const likedIds = ref<string[]>([]);
const dislikedIds = ref<string[]>([]);
const pendingRlEvents = ref<SearchActionEvent[]>([]);
const rlUserOverrideActive = ref(false);
const rlPrevSelectedArm = ref<string | null>(null);
const rlPrevIntentName = ref<string | null>(null);
const rlLastQueryText = ref("");
const rlLastApplied = ref(false);

/** draggable sidebar width */
const railW = ref(300);
const railMin = 220;
const railMax = 520;
const railCollapsed = ref(false);
const railDragging = ref(false);
let dragStartX = 0;
let dragStartW = 0;

const selectedDetail = ref<Record<string, unknown> | null>(null);

const resolvedQueryText = computed(() => {
  const m = data.value?.meta as Record<string, unknown> | undefined;
  return m && typeof m.query_text === "string" ? m.query_text : "";
});

const metaParsed = computed(() => {
  const m = data.value?.meta as Record<string, unknown> | undefined;
  return m?.parsed;
});

const metaPool = computed(() => {
  const m = data.value?.meta as Record<string, unknown> | undefined;
  if (!m) return "";
  const pr = m.pool_rows;
  const pk = m.pool_k;
  const rr = m.reranked;
  return `Pool: ${pr} rows (internal Top-${pk})${rr ? " · v2 re-ranked" : ""}`;
});

const rlBadgeText = computed(() => {
  const m = data.value?.meta as Record<string, unknown> | undefined;
  return m && typeof m.rl_strategy_label === "string" ? m.rl_strategy_label : "";
});

function startRailDrag(e: MouseEvent) {
  e.preventDefault();
  railDragging.value = true;
  dragStartX = e.clientX;
  dragStartW = railW.value;
  document.body.style.userSelect = "none";
  document.addEventListener("mousemove", onRailDrag);
  document.addEventListener("mouseup", endRailDrag);
}

function onRailDrag(e: MouseEvent) {
  if (!railDragging.value) return;
  const dx = e.clientX - dragStartX;
  railW.value = Math.min(railMax, Math.max(railMin, dragStartW + dx));
}

function endRailDrag() {
  railDragging.value = false;
  document.body.style.userSelect = "";
  document.removeEventListener("mousemove", onRailDrag);
  document.removeEventListener("mouseup", endRailDrag);
}

function onGlobalKey(e: KeyboardEvent) {
  if (e.key === "Escape") closeDetail();
}

onMounted(async () => {
  window.addEventListener("keydown", onGlobalKey);
  try {
    const r = await getStates();
    states.value = r.states.length ? r.states : ["PA"];
    if (!states.value.includes(browseState.value)) {
      browseState.value = states.value[0] ?? "PA";
    }
  } catch {
    states.value = ["PA", "NJ", "NV"];
  }
});

onUnmounted(() => {
  window.removeEventListener("keydown", onGlobalKey);
  endRailDrag();
});

function buildBody(discoverOnly: boolean, includePreferenceFeedback = true): SearchRequest {
  return {
    query: discoverOnly ? "" : nlQuery.value,
    state: browseState.value.trim().toUpperCase(),
    city: browseCity.value.trim() || null,
    top_k: topK.value,
    pool_k: poolK.value,
    keywords_extra: discoverOnly ? null : keywords.value.trim() || null,
    force_rebuild_index: forceRebuild.value,
    discover_only: discoverOnly,
    cuisines: discoverOnly ? [] : [...selectedCuisines.value],
    w_semantic: wSemantic.value,
    w_rating: wRating.value,
    w_price: wPrice.value,
    w_distance: wDistance.value,
    w_popularity: wPopularity.value,
    liked_business_ids: includePreferenceFeedback ? [...likedIds.value] : [],
    disliked_business_ids: includePreferenceFeedback ? [...dislikedIds.value] : [],
    rl_enabled: true,
    rl_user_overrode: rlUserOverrideActive.value,
    rl_prev_selected_arm: rlPrevSelectedArm.value,
    rl_prev_intent_name: rlPrevIntentName.value,
    rl_action_events: [...pendingRlEvents.value],
  };
}

async function runDiscover() {
  await run(buildBody(true));
  lastMode.value = "discover";
  step2Open.value = true;
}

async function runRefine() {
  await run(buildBody(false));
  lastMode.value = "refine";
}

async function run(body: SearchRequest) {
  err.value = null;
  loading.value = true;
  try {
    data.value = await postSearch(body);
    const meta = data.value.meta as Record<string, unknown>;
    pendingRlEvents.value = [];
    forceRebuild.value = false;
    rlPrevSelectedArm.value =
      typeof meta.rl_selected_arm === "string" ? meta.rl_selected_arm : null;
    rlPrevIntentName.value =
      typeof meta.rl_intent_name === "string" ? meta.rl_intent_name : null;
    rlLastQueryText.value = typeof meta.query_text === "string" ? meta.query_text : "";
    rlLastApplied.value = meta.rl_applied === true;
    rlUserOverrideActive.value = meta.rl_user_override_active === true;

    // When RL owns the ranking round, mirror the chosen preset back into the sliders.
    const weights = meta.rl_effective_weights;
    if (
      rlLastApplied.value &&
      weights &&
      typeof weights === "object" &&
      !Array.isArray(weights)
    ) {
      const record = weights as Record<string, unknown>;
      if (typeof record.w_semantic === "number") wSemantic.value = record.w_semantic;
      if (typeof record.w_rating === "number") wRating.value = record.w_rating;
      if (typeof record.w_price === "number") wPrice.value = record.w_price;
      if (typeof record.w_distance === "number") wDistance.value = record.w_distance;
      if (typeof record.w_popularity === "number") wPopularity.value = record.w_popularity;
    }
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

function queueRlEvent(action: SearchActionEvent["action"], businessId?: string) {
  if (
    action === "slider_override" &&
    pendingRlEvents.value.some((event) => event.action === "slider_override")
  ) {
    return;
  }
  pendingRlEvents.value = [
    ...pendingRlEvents.value,
    {
      action,
      business_id: businessId ?? null,
      query_text: rlLastQueryText.value || resolvedQueryText.value || null,
    },
  ];
}

function onSliderManualInput() {
  // The first manual slider move is the user's explicit opt-out from the RL preset.
  if (!rlLastApplied.value || rlUserOverrideActive.value || !rlPrevSelectedArm.value) return;
  rlUserOverrideActive.value = true;
  rlLastApplied.value = false;
  queueRlEvent("slider_override");
}

function toggleCuisine(c: string) {
  const i = selectedCuisines.value.indexOf(c);
  if (i >= 0) selectedCuisines.value = selectedCuisines.value.filter((x) => x !== c);
  else selectedCuisines.value = [...selectedCuisines.value, c];
}

function str(r: Record<string, unknown>, k: string): string {
  const v = r[k];
  if (v == null) return "";
  return String(v);
}

const FALLBACK_PHOTOS = [
  "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=640&h=400&fit=crop&q=80",
  "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=640&h=400&fit=crop&q=80",
  "https://images.unsplash.com/photo-1414235077428-338989a841e3?w=640&h=400&fit=crop&q=80",
  "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?w=640&h=400&fit=crop&q=80",
  "https://images.unsplash.com/photo-1552566626-52f8b828add9?w=640&h=400&fit=crop&q=80",
] as const;

function fallbackPhotoUrl(businessId: string): string {
  let h = 0;
  const s = businessId || "x";
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  const idx = Math.abs(h) % FALLBACK_PHOTOS.length;
  return FALLBACK_PHOTOS[idx] ?? FALLBACK_PHOTOS[0];
}

/** 列表缩略 + 弹层画廊 */
function photoUrls(row: Record<string, unknown>): string[] {
  const raw = row["photo_urls"];
  if (Array.isArray(raw) && raw.length) {
    return raw.map((x) => String(x)).filter(Boolean);
  }
  const u = str(row, "photo_url").trim();
  if (u) return [u];
  return [fallbackPhotoUrl(str(row, "business_id"))];
}

function thumbSrc(row: Record<string, unknown>): string {
  const urls = photoUrls(row);
  return urls[0] || fallbackPhotoUrl(str(row, "business_id"));
}

const GRAY_NO_PHOTO =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200"><rect width="100%" height="100%" fill="#1e293b"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#94a3b8" font-size="13">No photo</text></svg>'
  );

function onThumbErr(ev: Event, row: Record<string, unknown>) {
  const el = ev.target as HTMLImageElement;
  const bid = str(row, "business_id");
  if (el.dataset.fallback === "1") {
    el.src = GRAY_NO_PHOTO;
    return;
  }
  el.dataset.fallback = "1";
  el.src = fallbackPhotoUrl(`${bid}\0retry`);
}

function onGalleryImgErr(ev: Event) {
  const el = ev.target as HTMLImageElement;
  el.style.display = "none";
}

function openDetail(row: Record<string, unknown>) {
  selectedDetail.value = row;
  const bid = str(row, "business_id").trim();
  if (bid) queueRlEvent("detail_open", bid);
}

function closeDetail() {
  selectedDetail.value = null;
}

function num(r: Record<string, unknown>, k: string): number | null {
  const v = r[k];
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtPriceTier(tier: unknown): string {
  if (tier == null || tier === "") return "—";
  const t = Math.round(Number(tier));
  if (!Number.isFinite(t)) return "—";
  const x = Math.max(1, Math.min(4, t));
  return "$".repeat(x);
}

function distLabel(r: Record<string, unknown>): string {
  const d = num(r, "distance_km");
  if (d == null) return "—";
  return `${d.toFixed(1)} km`;
}

function scoreLine(r: Record<string, unknown>): string {
  const fs = num(r, "final_score");
  const sim = num(r, "similarity");
  const v2 = num(r, "v2_score");
  let s = `v1 ${fs != null ? fs.toFixed(2) : "—"} · sim ${sim != null ? sim.toFixed(2) : "—"}`;
  if (v2 != null) s += ` · v2 ${v2.toFixed(2)}`;
  return s;
}

function starBar(r: Record<string, unknown>): number {
  const s = num(r, "stars");
  if (s == null) return 0;
  return Math.min(1, Math.max(0, s / 5));
}

function toggleLike(bid: string) {
  const b = bid.trim();
  if (!b) return;
  if (likedIds.value.includes(b)) {
    likedIds.value = likedIds.value.filter((x) => x !== b);
  } else {
    likedIds.value = [...likedIds.value, b];
    dislikedIds.value = dislikedIds.value.filter((x) => x !== b);
    queueRlEvent("like", b);
  }
  void rerunFeedback();
}

function toggleDislike(bid: string) {
  const b = bid.trim();
  if (!b) return;
  if (dislikedIds.value.includes(b)) {
    dislikedIds.value = dislikedIds.value.filter((x) => x !== b);
  } else {
    dislikedIds.value = [...dislikedIds.value, b];
    likedIds.value = likedIds.value.filter((x) => x !== b);
  }
  void rerunFeedback();
}

function feedbackLabel(bid: string): string {
  if (likedIds.value.includes(bid)) return "Liked 👍";
  if (dislikedIds.value.includes(bid)) return "Disliked 👎";
  return "";
}

async function rerunFeedback() {
  if (!lastMode.value) return;
  await run(buildBody(lastMode.value === "discover"));
}

async function refreshResults() {
  if (!lastMode.value) return;
  queueRlEvent("refresh");
  rlUserOverrideActive.value = false;
  rlLastApplied.value = false;
  // Refresh asks RL for a new starting strategy, so skip the current v2 like/dislike rerank.
  await run(buildBody(lastMode.value === "discover", false));
}

function resetFeedback() {
  likedIds.value = [];
  dislikedIds.value = [];
  void rerunFeedback();
}
</script>

<template>
  <div class="app">
    <button
      v-if="railCollapsed"
      type="button"
      class="fab-open"
      @click="railCollapsed = false"
    >
      <span class="fab-icon">☰</span>
      Filters & Weights
    </button>

    <aside
      v-show="!railCollapsed"
      class="rail"
      :class="{ 'rail--drag': railDragging }"
      :style="{ width: railW + 'px' }"
    >
      <header class="rail-head">
        <div>
          <p class="rail-eyebrow">Search Parameters</p>
          <h2 class="rail-title">Location · Preferences · Sort</h2>
        </div>
        <button
          type="button"
          class="rail-collapse"
          title="Collapse sidebar"
          @click="railCollapsed = true"
        >
          ⟨
        </button>
      </header>

      <div class="rail-scroll">
        <section class="panel">
          <h3 class="h3">Step 1 — Location</h3>
          <label class="lbl">State</label>
          <select v-model="browseState" class="inp">
            <option v-for="s in states" :key="s" :value="s">{{ s }}</option>
          </select>
          <label class="lbl">City (optional, exact match)</label>
          <input v-model="browseCity" class="inp" type="text" placeholder="e.g. Philadelphia" />

          <button
            type="button"
            class="btn btn-primary"
            :disabled="loading || !browseState"
            @click="runDiscover"
          >
            Discover Nearby Restaurants
          </button>
        </section>

        <section class="panel">
          <h3 class="h3">Step 2 — Preferences & NL</h3>
          <details :open="step2Open" class="details">
            <summary>Natural language, cuisine, keywords</summary>
            <label class="lbl">Describe what you're looking for</label>
            <textarea
              v-model="nlQuery"
              class="inp area"
              rows="3"
              placeholder="e.g. cheap sushi near NYU within 3 km"
            />
            <span class="lbl">Cuisine</span>
            <div class="cuisine-grid">
              <label v-for="c in CUISINES" :key="c" class="cuisine"
                ><input
                  type="checkbox"
                  :checked="selectedCuisines.includes(c)"
                  @change="toggleCuisine(c)"
                />
                {{ c }}</label
              >
            </div>
            <label class="lbl">Extra keywords</label>
            <input v-model="keywords" class="inp" type="text" />
            <label class="lbl">Results count Top-K</label>
            <input v-model.number="topK" class="inp narrow" type="number" min="3" max="30" />
          </details>
          <button
            type="button"
            class="btn btn-secondary"
            :disabled="loading || !browseState"
            @click="runRefine"
          >
            Update by Preference
          </button>
        </section>

        <section class="panel panel-accent">
          <h3 class="h3">Sort Weights</h3>
          <p class="hint">Maps to the backend multi-factor <code>final_score</code>; drag the right edge to resize the sidebar.</p>
          <label class="chk"
            ><input v-model="forceRebuild" type="checkbox" /> Force rebuild TF-IDF index</label
          >

          <div class="rng">
            <div class="rng-h">
              <span>w_semantic text</span>
              <span class="rng-v">{{ wSemantic.toFixed(2) }}</span>
            </div>
            <input
              v-model.number="wSemantic"
              type="range"
              min="0"
              max="2"
              step="0.05"
              @input="onSliderManualInput"
            />
          </div>
          <div class="rng">
            <div class="rng-h">
              <span>w_rating stars</span>
              <span class="rng-v">{{ wRating.toFixed(2) }}</span>
            </div>
            <input
              v-model.number="wRating"
              type="range"
              min="0"
              max="2"
              step="0.05"
              @input="onSliderManualInput"
            />
          </div>
          <div class="rng">
            <div class="rng-h">
              <span>w_price price</span>
              <span class="rng-v">{{ wPrice.toFixed(2) }}</span>
            </div>
            <input
              v-model.number="wPrice"
              type="range"
              min="0"
              max="2"
              step="0.05"
              @input="onSliderManualInput"
            />
          </div>
          <div class="rng">
            <div class="rng-h">
              <span>w_distance distance</span>
              <span class="rng-v">{{ wDistance.toFixed(2) }}</span>
            </div>
            <input
              v-model.number="wDistance"
              type="range"
              min="0"
              max="2"
              step="0.05"
              @input="onSliderManualInput"
            />
          </div>
          <div class="rng">
            <div class="rng-h">
              <span>w_popularity popularity</span>
              <span class="rng-v">{{ wPopularity.toFixed(2) }}</span>
            </div>
            <input
              v-model.number="wPopularity"
              type="range"
              min="0"
              max="2"
              step="0.05"
              @input="onSliderManualInput"
            />
          </div>

          <h3 class="h3 mt1">v2 Candidate Pool</h3>
          <div class="rng">
            <div class="rng-h">
              <span>Pool size (for re-ranking)</span>
              <span class="rng-v">{{ poolK }}</span>
            </div>
            <input v-model.number="poolK" type="range" min="15" max="120" step="5" />
          </div>
        </section>
      </div>
    </aside>

    <div
      v-show="!railCollapsed"
      class="gutter"
      title="Drag to resize sidebar"
      @mousedown="startRailDrag"
    />

    <main class="main">
      <nav class="back">
        <router-link to="/" class="back-a">← Back to Home</router-link>
      </nav>

      <header class="hero">
        <h1 class="hero-title">Dining Intelligence</h1>
        <p class="hero-sub">
          Select a state/city for broad discovery, then refine with natural language and cuisine filters. The sidebar can be collapsed; click a card to view photos and details.
        </p>
        <p v-if="loading" class="status-pill">Searching…</p>
      </header>

      <p v-if="err" class="err-banner">{{ err }}</p>

      <div v-if="data && data.results.length" class="results-wrap">
        <div class="results-head">
          <h2>Recommended Results</h2>
          <p class="sub">{{ metaPool }}</p>
          <p v-if="rlBadgeText" class="rl-badge">{{ rlBadgeText }}</p>
          <details v-if="metaParsed" class="json-details">
            <summary>Parsed Rules</summary>
            <pre class="json-pre">{{ JSON.stringify(metaParsed, null, 2) }}</pre>
            <p v-if="resolvedQueryText" class="qt">query_text: <code>{{ resolvedQueryText }}</code></p>
          </details>
          <div class="results-tools">
            <button type="button" class="btn-ghost" :disabled="loading || !lastMode" @click="refreshResults">
              Refresh RL recommand
            </button>
            <button type="button" class="btn-ghost" @click="resetFeedback">Reset 👍/👎</button>
          </div>
        </div>

        <ul class="card-list">
          <li
            v-for="(row, i) in data.results"
            :key="str(row as Record<string, unknown>, 'business_id') + String(i)"
            class="r-card"
            role="button"
            tabindex="0"
            @click="openDetail(row as Record<string, unknown>)"
            @keydown.enter="openDetail(row as Record<string, unknown>)"
          >
            <div class="r-thumb">
              <img
                :src="thumbSrc(row as Record<string, unknown>)"
                :alt="str(row as Record<string, unknown>, 'name')"
                loading="lazy"
                referrerpolicy="no-referrer"
                @error="onThumbErr($event, row as Record<string, unknown>)"
              />
            </div>
            <div class="r-body">
              <div class="r-top">
                <span class="r-rank">#{{ i + 1 }}</span>
                <h3 class="r-name">{{ str(row as Record<string, unknown>, "name") }}</h3>
                <p class="r-meta">
                  {{ str(row as Record<string, unknown>, "city") }},
                  {{ str(row as Record<string, unknown>, "state") }}
                  · {{ num(row as Record<string, unknown>, "stars")?.toFixed(1) ?? "—" }}★
                  · {{ str(row as Record<string, unknown>, "review_count") }} reviews
                </p>
                <p class="r-dim">
                  {{ fmtPriceTier((row as Record<string, unknown>).price_tier) }} ·
                  {{ distLabel(row as Record<string, unknown>) }} ·
                  {{ scoreLine(row as Record<string, unknown>) }}
                </p>
              </div>
              <div
                v-if="str(row as Record<string, unknown>, 'business_id')"
                class="r-actions"
                @click.stop
              >
                <button
                  type="button"
                  class="fb"
                  @click="toggleLike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  👍
                </button>
                <button
                  type="button"
                  class="fb"
                  @click="toggleDislike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  👎
                </button>
                <span v-if="feedbackLabel(str(row as Record<string, unknown>, 'business_id'))" class="fb-t">{{
                  feedbackLabel(str(row as Record<string, unknown>, "business_id"))
                }}</span>
                <span class="tap-hint">Click card to view photos</span>
              </div>
              <div class="bar" @click.stop>
                <div
                  class="bar-fill"
                  :style="{ width: starBar(row as Record<string, unknown>) * 100 + '%' }"
                />
              </div>
            </div>
          </li>
        </ul>
      </div>

      <p v-else-if="data" class="empty">No results found. Try a different state/city or broaden your filters.</p>

      <p class="foot">TF-IDF cosine + multi-factor re-ranking, consistent with <code>recommend_keywords</code></p>
    </main>

    <Teleport to="body">
      <div
        v-if="selectedDetail"
        class="modal-root"
        role="presentation"
        @keydown.escape.prevent="closeDetail"
      >
        <div class="modal-back" @click="closeDetail" />
        <div class="modal-box" role="dialog" aria-modal="true" aria-label="Restaurant Details" @click.stop>
          <button type="button" class="modal-x" aria-label="Close" @click="closeDetail">×</button>
          <h2 class="modal-title">{{ str(selectedDetail, "name") }}</h2>
          <p class="modal-sub">
            {{ str(selectedDetail, "address") }} · {{ str(selectedDetail, "city") }},
            {{ str(selectedDetail, "state") }}
          </p>
          <p class="modal-sub2">
            {{ str(selectedDetail, "categories") }}
          </p>
          <div class="gallery">
            <figure v-for="(u, gi) in photoUrls(selectedDetail)" :key="gi" class="g-fig">
              <img
                :src="u"
                :alt="`Photo ${gi + 1}`"
                loading="lazy"
                referrerpolicy="no-referrer"
                @error="onGalleryImgErr"
              />
            </figure>
          </div>
          <p class="modal-score">
            <strong>{{ num(selectedDetail, "stars")?.toFixed(1) ?? "—" }}</strong> stars ·
            {{ str(selectedDetail, "review_count") }} reviews ·
            {{ fmtPriceTier(selectedDetail["price_tier"]) }} · {{ distLabel(selectedDetail) }}
          </p>
          <div v-if="str(selectedDetail, 'business_id')" class="modal-actions">
            <button
              type="button"
              class="btn btn-primary"
              @click="toggleLike(str(selectedDetail, 'business_id'))"
            >
              👍 Like
            </button>
            <button
              type="button"
              class="btn btn-secondary"
              @click="toggleDislike(str(selectedDetail, 'business_id'))"
            >
              👎 Dislike
            </button>
            <code class="bid">ID: {{ str(selectedDetail, "business_id") }}</code>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.app {
  --bg: #0b1220;
  --panel: #111827;
  --panel-2: #0f172a;
  --border: rgba(148, 163, 184, 0.18);
  --text: #e2e8f0;
  --muted: #94a3b8;
  --accent: #818cf8;
  --accent-2: #6366f1;
  min-height: 100vh;
  display: flex;
  background: linear-gradient(160deg, #0b1220 0%, #0f172a 45%, #111827 100%);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
  font-size: 15px;
}

.fab-open {
  position: fixed;
  z-index: 5;
  left: 0.75rem;
  top: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.9rem;
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: 999px;
  font-size: 0.9rem;
  cursor: pointer;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.fab-icon {
  font-size: 1.1rem;
  opacity: 0.9;
}

.rail {
  flex: 0 0 auto;
  min-width: 220px;
  max-width: 520px;
  width: 300px;
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  z-index: 2;
  box-shadow: 8px 0 40px rgba(0, 0, 0, 0.2);
}
.rail--drag {
  user-select: none;
}
.rail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 1.1rem 1rem 0.75rem;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.rail-eyebrow {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 0 0 0.2rem;
}
.rail-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 700;
}
.rail-collapse {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--border);
  background: var(--panel-2);
  color: var(--muted);
  border-radius: 8px;
  cursor: pointer;
  font-size: 1.1rem;
  line-height: 1;
}
.rail-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 0.75rem 0.9rem 1.5rem;
  scrollbar-gutter: stable;
}

.gutter {
  width: 6px;
  flex: 0 0 6px;
  background: linear-gradient(90deg, rgba(0, 0, 0, 0.15), transparent);
  cursor: col-resize;
  z-index: 3;
}
.gutter:hover {
  background: rgba(129, 140, 248, 0.3);
}

.panel {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 0.85rem 1rem;
  margin-bottom: 0.85rem;
}
.panel-accent {
  background: linear-gradient(145deg, #1a2040 0%, #111827 100%);
  border-color: rgba(99, 102, 241, 0.25);
}
.h3 {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--accent);
  margin: 0 0 0.6rem;
}
.lbl {
  display: block;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--muted);
  margin: 0.5rem 0 0.25rem;
}
.inp {
  width: 100%;
  padding: 0.45rem 0.5rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: #0b0f1a;
  color: var(--text);
  font: inherit;
  box-sizing: border-box;
}
.inp:focus {
  outline: 1px solid var(--accent-2);
}
.area {
  min-height: 3.2rem;
  resize: vertical;
}
.narrow {
  max-width: 8rem;
}
.cuisine-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.5rem;
  margin: 0.35rem 0 0.5rem;
  font-size: 0.82rem;
}
.cuisine {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  cursor: pointer;
  color: #cbd5e1;
}
.details {
  margin-bottom: 0.65rem;
}
.details summary {
  cursor: pointer;
  color: var(--muted);
  font-size: 0.86rem;
  margin-bottom: 0.4rem;
}

.btn {
  width: 100%;
  margin-top: 0.65rem;
  padding: 0.5rem 0.75rem;
  border: none;
  border-radius: 9px;
  font-weight: 600;
  cursor: pointer;
  font: inherit;
}
.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent-2) 0%, #4f46e5 100%);
  color: #fff;
  margin-top: 0.4rem;
}
.btn-secondary {
  background: #334155;
  color: #f1f5f9;
}
.hint {
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0 0 0.5rem;
}
.hint code {
  font-size: 0.72rem;
  color: #a5b4fc;
  background: rgba(99, 102, 241, 0.1);
  padding: 0.05em 0.2em;
  border-radius: 3px;
}
.chk {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 0.4rem 0 0.75rem;
  color: #cbd5e1;
  font-size: 0.82rem;
  cursor: pointer;
}
.rng {
  margin-bottom: 0.4rem;
}
.rng input[type="range"] {
  width: 100%;
  accent-color: var(--accent-2);
  margin-top: 0.1rem;
}
.rng-h {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  color: var(--muted);
}
.rng-v {
  color: #a5b4fc;
  font-weight: 600;
}
.mt1 {
  margin-top: 0.6rem;
}

.main {
  flex: 1 1 0;
  min-width: 0;
  align-self: stretch;
  width: 100%;
  max-width: none;
  padding: 1.25rem 1.4rem 2.5rem;
  box-sizing: border-box;
}
.back {
  margin-bottom: 0.25rem;
}
.back-a {
  color: #a5b4fc;
  text-decoration: none;
  font-size: 0.88rem;
}
.back-a:hover {
  text-decoration: underline;
}
.hero {
  margin-bottom: 1.25rem;
}
.hero-title {
  margin: 0.25rem 0 0.35rem;
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(100deg, #e2e8f0, #a5b4fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.55;
  margin: 0 0 0.5rem;
  max-width: 36rem;
}
.status-pill {
  display: inline-block;
  margin: 0;
  padding: 0.2rem 0.55rem;
  background: rgba(99, 102, 241, 0.15);
  color: #a5b4fc;
  border-radius: 999px;
  font-size: 0.78rem;
}

.err-banner {
  background: rgba(127, 29, 29, 0.3);
  border: 1px solid rgba(248, 113, 113, 0.3);
  color: #fecaca;
  padding: 0.65rem 0.85rem;
  border-radius: 10px;
  margin: 0.5rem 0 1rem;
  font-size: 0.88rem;
}

.results-head {
  margin-bottom: 1rem;
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 0.9rem;
}
.results-head h2 {
  margin: 0;
  font-size: 1.1rem;
}
.sub {
  margin: 0;
  color: var(--muted);
  font-size: 0.86rem;
}
.rl-badge {
  margin: 0;
  padding: 0.28rem 0.6rem;
  border-radius: 999px;
  background: rgba(99, 102, 241, 0.18);
  border: 1px solid rgba(129, 140, 248, 0.3);
  color: #c7d2fe;
  font-size: 0.8rem;
}
.results-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-left: auto;
}
.json-details {
  flex-basis: 100%;
  margin-top: 0.25rem;
  font-size: 0.8rem;
  color: var(--muted);
}
.json-pre {
  background: #0b0f1a;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 0.6rem;
  font-size: 0.7rem;
  max-height: 8rem;
  overflow: auto;
  margin: 0.3rem 0 0.25rem;
}
.qt {
  font-size: 0.8rem;
  color: #cbd5e1;
}
.btn-ghost {
  background: transparent;
  border: 1px dashed rgba(148, 163, 184, 0.35);
  color: var(--muted);
  font-size: 0.8rem;
  border-radius: 6px;
  padding: 0.25rem 0.5rem;
  cursor: pointer;
}
.btn-ghost:hover {
  border-color: var(--accent);
  color: #e2e8f0;
}
.btn-ghost:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.card-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}
@media (min-width: 900px) {
  .card-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
    gap: 0.9rem;
    align-content: start;
  }
}

.r-card {
  display: flex;
  gap: 0;
  background: #111827;
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
  transition: box-shadow 0.2s, border-color 0.2s, transform 0.2s;
  cursor: pointer;
  outline: none;
}
.r-card:hover {
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
  border-color: rgba(99, 102, 241, 0.35);
  transform: translateY(-1px);
}
.r-card:focus-visible {
  box-shadow: 0 0 0 2px #6366f1;
}
.r-thumb {
  flex: 0 0 132px;
  min-height: 100px;
  max-height: 150px;
  background: #0b0f1a;
  overflow: hidden;
}
.r-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.r-body {
  flex: 1;
  min-width: 0;
  padding: 0.7rem 0.95rem 0.75rem;
  display: flex;
  flex-direction: column;
}
.r-top {
  flex: 1;
}
.r-rank {
  font-size: 0.72rem;
  color: var(--accent);
  font-weight: 700;
  margin-right: 0.3rem;
}
.r-name {
  display: inline;
  font-size: 1.02rem;
  font-weight: 700;
  margin: 0;
  color: #f8fafc;
}
.r-meta {
  font-size: 0.8rem;
  color: #cbd5e1;
  margin: 0.2rem 0 0.15rem;
}
.r-dim {
  font-size: 0.76rem;
  color: var(--muted);
  margin: 0 0 0.35rem;
  line-height: 1.35;
  word-break: break-word;
}
.r-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem 0.6rem;
  margin-top: 0.15rem;
}
.fb {
  padding: 0.2rem 0.45rem;
  font-size: 0.8rem;
  border: 1px solid var(--border);
  background: #0b0f1a;
  color: #e2e8f0;
  border-radius: 6px;
  cursor: pointer;
  line-height: 1.2;
}
.fb:hover {
  background: #1e293b;
}
.fb-t {
  font-size: 0.72rem;
  color: #a5b4fc;
}
.tap-hint {
  font-size: 0.7rem;
  color: #64748b;
  margin-left: 0.15rem;
}
@media (max-width: 600px) {
  .r-card {
    flex-direction: column;
  }
  .r-thumb {
    flex: none;
    max-height: 200px;
    min-height: 150px;
  }
}

.bar {
  height: 4px;
  background: #1e293b;
  border-radius: 4px;
  margin-top: 0.45rem;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #f59e0b, #fbbf24);
  border-radius: 4px;
  transition: width 0.25s ease;
}

.empty {
  color: var(--muted);
  text-align: center;
  padding: 2.5rem 0.5rem;
}
.foot {
  margin-top: 1.5rem;
  color: #64748b;
  font-size: 0.78rem;
  max-width: 36rem;
}
.foot code {
  font-size: 0.72rem;
  color: #a5b4fc;
}

/* modal */
.modal-root {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
  box-sizing: border-box;
}
.modal-back {
  position: absolute;
  inset: 0;
  background: rgba(2, 6, 23, 0.8);
  backdrop-filter: blur(4px);
}
.modal-box {
  position: relative;
  z-index: 1;
  max-width: min(700px, 100%);
  max-height: min(88vh, 100%);
  overflow: auto;
  width: 100%;
  background: #111827;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.35rem 1.25rem 1.1rem;
  box-shadow: 0 32px 80px rgba(0, 0, 0, 0.5);
  animation: in 0.2s ease;
}
@keyframes in {
  from {
    transform: scale(0.98);
    opacity: 0.8;
  }
  to {
    transform: none;
    opacity: 1;
  }
}
.modal-x {
  position: absolute;
  right: 0.65rem;
  top: 0.55rem;
  width: 2.1rem;
  height: 2.1rem;
  border: none;
  background: #1e293b;
  color: #94a3b8;
  border-radius: 8px;
  font-size: 1.25rem;
  line-height: 1;
  cursor: pointer;
}
.modal-x:hover {
  background: #334155;
  color: #e2e8f0;
}
.modal-title {
  margin: 0 2.25rem 0.35rem 0;
  font-size: 1.35rem;
  font-weight: 800;
  line-height: 1.2;
  color: #f8fafc;
}
.modal-sub,
.modal-sub2 {
  color: #94a3b8;
  font-size: 0.9rem;
  line-height: 1.4;
  margin: 0 0 0.3rem;
}
.modal-sub2 {
  font-size: 0.82rem;
  margin-bottom: 0.85rem;
  word-break: break-word;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 0.5rem;
  margin: 0.2rem 0 1rem;
}
.g-fig {
  margin: 0;
  border-radius: 10px;
  overflow: hidden;
  aspect-ratio: 4/3;
  background: #0b0f1a;
  border: 1px solid var(--border);
}
.g-fig img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.modal-score {
  font-size: 0.9rem;
  color: #cbd5e1;
  margin: 0 0 0.8rem;
}
.modal-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.6rem;
}
.modal-actions .btn {
  width: auto;
  margin: 0;
  padding: 0.4rem 0.85rem;
  font-size: 0.88rem;
}
.bid {
  font-size: 0.72rem;
  color: #64748b;
  word-break: break-all;
}
</style>
