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

/** Resizable filter rail width (px) */
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
  return `Pool ${pr} rows (top ${pk} internally)${rr ? " · v2 re-ranked" : ""}`;
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

/** Thumbnail list + modal gallery */
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
  if (likedIds.value.includes(bid)) return "Liked";
  if (dislikedIds.value.includes(bid)) return "Passed";
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
  <div class="search-layout">
    <button
      v-if="railCollapsed"
      type="button"
      class="fab-open"
      @click="railCollapsed = false"
    >
      <span class="fab-icon">☰</span>
      Filters & ranking
    </button>

    <aside
      v-show="!railCollapsed"
      class="rail"
      :class="{ 'rail--drag': railDragging }"
      :style="{ width: railW + 'px' }"
    >
      <header class="rail-head">
        <div>
          <p class="rail-eyebrow">Search controls</p>
          <h2 class="rail-title">Location · taste · weights</h2>
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
          <h3 class="h3">Step 1 — Where</h3>
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
            Browse top spots
          </button>
        </section>

        <section class="panel">
          <h3 class="h3">Step 2 — Refine</h3>
          <details :open="step2Open" class="details">
            <summary>Natural language, cuisines, keywords</summary>
            <label class="lbl">What are you in the mood for?</label>
            <textarea
              v-model="nlQuery"
              class="inp area"
              rows="3"
              placeholder="e.g. affordable sushi near NYU, within 3 km"
            />
            <span class="lbl">Cuisines</span>
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
            <label class="lbl">Results (top-K)</label>
            <input v-model.number="topK" class="inp narrow" type="number" min="3" max="30" />
          </details>
          <button
            type="button"
            class="btn btn-secondary"
            :disabled="loading || !browseState"
            @click="runRefine"
          >
            Search with preferences
          </button>
        </section>

        <section class="panel panel-accent">
          <h3 class="h3">Ranking weights</h3>
          <p class="hint">
            Maps to backend multi-signal <code>final_score</code>. Drag the gutter to resize this panel.
          </p>
          <label class="chk"
            ><input v-model="forceRebuild" type="checkbox" /> Force rebuild TF-IDF index</label
          >

          <div class="rng">
            <div class="rng-h">
              <span>Semantic (text)</span>
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
              <span>Rating (stars)</span>
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
              <span>Price fit</span>
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
              <span>Distance</span>
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
              <span>Popularity</span>
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

          <h3 class="h3 mt1">Re-rank pool</h3>
          <div class="rng">
            <div class="rng-h">
              <span>Pool size (for likes / dislikes)</span>
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

    <main class="search-main">
      <nav class="back">
        <router-link to="/" class="back-a">← Home</router-link>
      </nav>

      <header class="hero">
        <p class="hero-kicker">Curated for your city</p>
        <h1 class="hero-title">Find your next favorite</h1>
        <p class="hero-sub">
          Start with state and optional city, then refine with language and cuisines. Collapse the rail
          for more space; tap a card for photos and details.
        </p>
        <p v-if="loading" class="status-pill">Searching…</p>
      </header>

      <p v-if="err" class="err-banner">{{ err }}</p>

      <div v-if="data && data.results.length" class="results-wrap">
        <div class="results-head">
          <div class="results-head-text">
            <h2>Places for you</h2>
            <p class="sub">{{ metaPool }}</p>
            <p v-if="rlBadgeText" class="rl-badge">{{ rlBadgeText }}</p>
          </div>
          <div class="results-tools">
            <button type="button" class="btn-ghost" :disabled="loading || !lastMode" @click="refreshResults">
              New bandit draw
            </button>
            <button type="button" class="btn-ghost" @click="resetFeedback">Clear likes / passes</button>
          </div>
          <details v-if="metaParsed" class="json-details">
            <summary>Parsed query</summary>
            <pre class="json-pre">{{ JSON.stringify(metaParsed, null, 2) }}</pre>
            <p v-if="resolvedQueryText" class="qt">query_text: <code>{{ resolvedQueryText }}</code></p>
          </details>
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
              <span class="r-rank">#{{ i + 1 }}</span>
            </div>
            <div class="r-body">
              <div class="r-top">
                <h3 class="r-name">{{ str(row as Record<string, unknown>, "name") }}</h3>
                <p class="r-meta">
                  {{ str(row as Record<string, unknown>, "city") }},
                  {{ str(row as Record<string, unknown>, "state") }}
                  · {{ num(row as Record<string, unknown>, "stars")?.toFixed(1) ?? "—" }} ★
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
                  class="fb fb-like"
                  @click="toggleLike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  Like
                </button>
                <button
                  type="button"
                  class="fb fb-pass"
                  @click="toggleDislike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  Pass
                </button>
                <span v-if="feedbackLabel(str(row as Record<string, unknown>, 'business_id'))" class="fb-t">{{
                  feedbackLabel(str(row as Record<string, unknown>, "business_id"))
                }}</span>
                <span class="tap-hint">Elsewhere on card opens gallery</span>
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

      <p v-else-if="data" class="empty">No venues match. Try another state or city, or relax filters.</p>

      <p class="foot">
        TF-IDF cosine similarity + multi-signal blend (<code>recommend_keywords</code>).
      </p>
    </main>

    <Teleport to="body">
      <div
        v-if="selectedDetail"
        class="modal-root"
        role="presentation"
        @keydown.escape.prevent="closeDetail"
      >
        <div class="modal-back" @click="closeDetail" />
        <div class="modal-box" role="dialog" aria-modal="true" aria-label="Restaurant details" @click.stop>
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
              Like
            </button>
            <button
              type="button"
              class="btn btn-secondary"
              @click="toggleDislike(str(selectedDetail, 'business_id'))"
            >
              Pass
            </button>
            <code class="bid">ID: {{ str(selectedDetail, "business_id") }}</code>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.search-layout {
  --ink: #1c1917;
  --muted: #78716c;
  --line: #e7e5e4;
  --panel: #ffffff;
  --panel-2: #fafaf9;
  --accent: #dc2626;
  --accent-2: #b91c1c;
  --accent-soft: #fef2f2;
  --shadow: 0 4px 24px rgba(28, 25, 23, 0.08);
  --shadow-lg: 0 20px 50px rgba(28, 25, 23, 0.12);
  min-height: calc(100vh - 0px);
  display: flex;
  background: linear-gradient(180deg, #f5f5f4 0%, #fafaf9 40%, #ffffff 100%);
  color: var(--ink);
  font-family: "Plus Jakarta Sans", system-ui, sans-serif;
  font-size: 15px;
}

.fab-open {
  position: fixed;
  z-index: 40;
  left: 0.85rem;
  top: 4.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 1rem;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 999px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--shadow-lg);
}
.fab-icon {
  font-size: 1.1rem;
  opacity: 0.85;
}

.rail {
  flex: 0 0 auto;
  min-width: 220px;
  max-width: 520px;
  width: 300px;
  background: var(--panel);
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  z-index: 2;
  box-shadow: 4px 0 32px rgba(28, 25, 23, 0.04);
}
.rail--drag {
  user-select: none;
}
.rail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 1rem 1rem 0.85rem;
  border-bottom: 1px solid var(--line);
  flex-shrink: 0;
  background: linear-gradient(180deg, #fffefb 0%, #fff 100%);
}
.rail-eyebrow {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 700;
  color: #a8a29e;
  margin: 0 0 0.2rem;
}
.rail-title {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.25;
}
.rail-collapse {
  flex-shrink: 0;
  width: 2.1rem;
  height: 2.1rem;
  border: 1px solid var(--line);
  background: #fafaf9;
  color: #57534e;
  border-radius: 10px;
  cursor: pointer;
  font-size: 1.1rem;
  line-height: 1;
  transition: background 0.15s, border-color 0.15s;
}
.rail-collapse:hover {
  background: var(--accent-soft);
  border-color: #fecaca;
  color: var(--accent-2);
}
.rail-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 0.85rem 0.9rem 1.5rem;
  scrollbar-gutter: stable;
}

.gutter {
  width: 6px;
  flex: 0 0 6px;
  background: linear-gradient(90deg, rgba(231, 229, 228, 0.9), transparent);
  cursor: col-resize;
  z-index: 3;
}
.gutter:hover {
  background: linear-gradient(90deg, rgba(248, 113, 113, 0.45), transparent);
}

.panel {
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.95rem 0.9rem 1.05rem;
  margin-bottom: 0.85rem;
}
.panel-accent {
  background: linear-gradient(160deg, #fffafa 0%, #fafaf9 100%);
  border-color: #fecaca;
}
.h3 {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 800;
  color: var(--accent-2);
  margin: 0 0 0.55rem;
}
.lbl {
  display: block;
  font-size: 0.78rem;
  font-weight: 700;
  color: #57534e;
  margin: 0.5rem 0 0.25rem;
}
.inp {
  width: 100%;
  padding: 0.5rem 0.6rem;
  border-radius: 10px;
  border: 1px solid #d6d3d1;
  background: #fff;
  color: var(--ink);
  font: inherit;
  box-sizing: border-box;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.inp:focus {
  outline: none;
  border-color: #f87171;
  box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.22);
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
  gap: 0.35rem 0.45rem;
  margin: 0.35rem 0 0.5rem;
  font-size: 0.8rem;
}
.cuisine {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  cursor: pointer;
  color: #44403c;
  font-weight: 500;
}
.details {
  margin-bottom: 0.65rem;
}
.details summary {
  cursor: pointer;
  color: var(--muted);
  font-size: 0.84rem;
  font-weight: 600;
  margin-bottom: 0.45rem;
}

.btn {
  width: 100%;
  margin-top: 0.65rem;
  padding: 0.55rem 0.85rem;
  border: none;
  border-radius: 999px;
  font-weight: 700;
  cursor: pointer;
  font: inherit;
  font-size: 0.875rem;
  transition: transform 0.15s, box-shadow 0.15s;
}
.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.btn-primary {
  background: linear-gradient(135deg, #b91c1c, #ef4444);
  color: #fff;
  margin-top: 0.45rem;
  box-shadow: 0 4px 14px rgba(220, 38, 38, 0.3);
}
.btn-primary:not(:disabled):hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(185, 28, 28, 0.38);
}
.btn-secondary {
  background: #fff;
  color: var(--ink);
  border: 1px solid var(--line);
}
.btn-secondary:not(:disabled):hover {
  border-color: #d6d3d1;
  background: #fafaf9;
}
.hint {
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0 0 0.5rem;
}
.hint code {
  font-size: 0.72rem;
  color: #991b1b;
  background: var(--accent-soft);
  padding: 0.08em 0.28em;
  border-radius: 4px;
}
.chk {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0.4rem 0 0.75rem;
  color: #44403c;
  font-size: 0.82rem;
  cursor: pointer;
  font-weight: 500;
}
.rng {
  margin-bottom: 0.45rem;
}
.rng input[type="range"] {
  width: 100%;
  accent-color: var(--accent);
  margin-top: 0.12rem;
}
.rng-h {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  color: var(--muted);
  font-weight: 600;
}
.rng-v {
  color: var(--accent-2);
  font-weight: 800;
}
.mt1 {
  margin-top: 0.65rem;
}

.search-main {
  flex: 1 1 0;
  min-width: 0;
  align-self: stretch;
  width: 100%;
  max-width: none;
  padding: 1.35rem 1.5rem 2.75rem;
  box-sizing: border-box;
}
.back {
  margin-bottom: 0.35rem;
}
.back-a {
  color: var(--accent);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 700;
}
.back-a:hover {
  text-decoration: underline;
}
.hero {
  margin-bottom: 1.35rem;
}
.hero-kicker {
  margin: 0 0 0.35rem;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #a8a29e;
}
.hero-title {
  margin: 0 0 0.45rem;
  font-size: clamp(1.65rem, 3.5vw, 2.1rem);
  font-weight: 800;
  letter-spacing: -0.035em;
  line-height: 1.15;
  color: var(--ink);
}
.hero-sub {
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.55;
  margin: 0 0 0.5rem;
  max-width: 40rem;
}
.status-pill {
  display: inline-block;
  margin: 0.35rem 0 0;
  padding: 0.28rem 0.75rem;
  background: var(--accent-soft);
  color: var(--accent-2);
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 700;
}

.err-banner {
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  padding: 0.65rem 0.9rem;
  border-radius: 12px;
  margin: 0.5rem 0 1rem;
  font-size: 0.88rem;
  font-weight: 500;
}

.results-head {
  margin-bottom: 1.25rem;
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 0.75rem 1rem;
}
.results-head-text {
  flex: 1;
  min-width: 12rem;
}
.results-head h2 {
  margin: 0 0 0.35rem;
  font-size: 1.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.sub {
  margin: 0;
  color: var(--muted);
  font-size: 0.86rem;
}
.rl-badge {
  margin: 0.5rem 0 0;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  color: #991b1b;
  font-size: 0.8rem;
  font-weight: 600;
  display: inline-block;
}
.results-tools {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
}
.json-details {
  flex-basis: 100%;
  margin-top: 0.35rem;
  font-size: 0.8rem;
  color: var(--muted);
}
.json-pre {
  background: #1c1917;
  color: #e7e5e4;
  border-radius: 12px;
  padding: 0.55rem 0.65rem;
  font-size: 0.7rem;
  max-height: 8rem;
  overflow: auto;
  margin: 0.35rem 0 0.25rem;
}
.qt {
  font-size: 0.8rem;
  color: #57534e;
}
.btn-ghost {
  background: #fff;
  border: 1px solid var(--line);
  color: #57534e;
  font-size: 0.8rem;
  font-weight: 600;
  border-radius: 999px;
  padding: 0.35rem 0.85rem;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s;
}
.btn-ghost:hover:not(:disabled) {
  border-color: #fca5a5;
  color: var(--accent-2);
}
.btn-ghost:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.card-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
  gap: 1.25rem;
  align-content: start;
}

.r-card {
  display: flex;
  flex-direction: column;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 20px;
  overflow: hidden;
  transition: box-shadow 0.2s, border-color 0.2s, transform 0.2s;
  cursor: pointer;
  outline: none;
  box-shadow: var(--shadow);
}
.r-card:hover {
  box-shadow: var(--shadow-lg);
  border-color: #fecaca;
  transform: translateY(-3px);
}
.r-card:focus-visible {
  box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.45);
}
.r-thumb {
  position: relative;
  aspect-ratio: 16 / 10;
  background: #f5f5f4;
  overflow: hidden;
}
.r-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.r-rank {
  position: absolute;
  left: 0.65rem;
  top: 0.65rem;
  font-size: 0.72rem;
  font-weight: 800;
  color: #fff;
  background: rgba(28, 25, 23, 0.72);
  backdrop-filter: blur(6px);
  padding: 0.22rem 0.5rem;
  border-radius: 999px;
  letter-spacing: 0.02em;
}
.r-body {
  flex: 1;
  min-width: 0;
  padding: 0.95rem 1rem 1rem;
  display: flex;
  flex-direction: column;
}
.r-top {
  flex: 1;
}
.r-name {
  font-size: 1.05rem;
  font-weight: 800;
  margin: 0 0 0.35rem;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1.25;
}
.r-meta {
  font-size: 0.82rem;
  color: #57534e;
  margin: 0 0 0.35rem;
  line-height: 1.4;
}
.r-dim {
  font-size: 0.74rem;
  color: #a8a29e;
  margin: 0 0 0.5rem;
  line-height: 1.4;
  word-break: break-word;
}
.r-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.55rem;
  margin-top: auto;
}
.fb {
  padding: 0.35rem 0.75rem;
  font-size: 0.78rem;
  font-weight: 700;
  border: 1px solid var(--line);
  background: #fafaf9;
  color: var(--ink);
  border-radius: 999px;
  cursor: pointer;
  line-height: 1.2;
  transition: background 0.15s, border-color 0.15s;
}
.fb-like:hover {
  background: #fef2f2;
  border-color: #fecaca;
  color: #b91c1c;
}
.fb-pass:hover {
  background: #fef2f2;
  border-color: #fca5a5;
  color: #991b1b;
}
.fb-t {
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--accent-2);
}
.tap-hint {
  font-size: 0.68rem;
  color: #a8a29e;
  margin-left: 0.1rem;
  flex-basis: 100%;
}

.bar {
  height: 4px;
  background: #e7e5e4;
  border-radius: 4px;
  margin-top: 0.65rem;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #b91c1c, #f87171);
  border-radius: 4px;
  transition: width 0.25s ease;
}

.empty {
  color: var(--muted);
  text-align: center;
  padding: 2.75rem 1rem;
  font-weight: 500;
}
.foot {
  margin-top: 2rem;
  color: #a8a29e;
  font-size: 0.78rem;
  max-width: 38rem;
  line-height: 1.5;
}
.foot code {
  font-size: 0.72rem;
  color: #78716c;
  background: #f5f5f4;
  padding: 0.08em 0.25em;
  border-radius: 4px;
}

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
  background: rgba(28, 25, 23, 0.55);
  backdrop-filter: blur(6px);
}
.modal-box {
  position: relative;
  z-index: 1;
  max-width: min(720px, 100%);
  max-height: min(90vh, 100%);
  overflow: auto;
  width: 100%;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1.35rem 1.25rem 1.15rem;
  box-shadow: var(--shadow-lg);
  animation: modalIn 0.22s ease;
}
@keyframes modalIn {
  from {
    transform: scale(0.98);
    opacity: 0.85;
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
  width: 2.15rem;
  height: 2.15rem;
  border: 1px solid var(--line);
  background: #fafaf9;
  color: #57534e;
  border-radius: 10px;
  font-size: 1.2rem;
  line-height: 1;
  cursor: pointer;
}
.modal-x:hover {
  background: #f5f5f4;
  color: var(--ink);
}
.modal-title {
  margin: 0 2.5rem 0.4rem 0;
  font-size: 1.35rem;
  font-weight: 800;
  line-height: 1.2;
  color: var(--ink);
  letter-spacing: -0.02em;
}
.modal-sub,
.modal-sub2 {
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.45;
  margin: 0 0 0.3rem;
}
.modal-sub2 {
  font-size: 0.82rem;
  margin-bottom: 0.9rem;
  word-break: break-word;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 0.55rem;
  margin: 0.2rem 0 1rem;
}
.g-fig {
  margin: 0;
  border-radius: 12px;
  overflow: hidden;
  aspect-ratio: 4/3;
  background: #f5f5f4;
  border: 1px solid var(--line);
}
.g-fig img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.modal-score {
  font-size: 0.9rem;
  color: #57534e;
  margin: 0 0 0.85rem;
  font-weight: 500;
}
.modal-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem 0.65rem;
}
.modal-actions .btn {
  width: auto;
  margin: 0;
  padding: 0.45rem 1rem;
  font-size: 0.875rem;
}
.modal-actions .btn-primary {
  margin: 0;
}
.bid {
  font-size: 0.72rem;
  color: #a8a29e;
  word-break: break-all;
}
</style>
