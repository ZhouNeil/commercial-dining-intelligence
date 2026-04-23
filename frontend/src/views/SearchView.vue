<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  getStates,
  postSearch,
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
const step2Open = ref(false);

const loading = ref(false);
const err = ref<string | null>(null);
const data = ref<SearchResponse | null>(null);
const lastMode = ref<"discover" | "refine" | null>(null);

const likedIds = ref<string[]>([]);
const dislikedIds = ref<string[]>([]);

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
  return `候选池 ${pr} 行（内部 Top-${pk}）${rr ? " · 已按 👍/👎 v2 重排" : ""}`;
});

onMounted(async () => {
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

function buildBody(discoverOnly: boolean): SearchRequest {
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
    liked_business_ids: [...likedIds.value],
    disliked_business_ids: [...dislikedIds.value],
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
    forceRebuild.value = false;
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
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

/** 与 backend/services/retrieval_service 占位列表一致（CDN 失败时二次回退） */
const FALLBACK_PHOTOS = [
  "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=320&h=200&fit=crop&q=80",
  "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=320&h=200&fit=crop&q=80",
  "https://images.unsplash.com/photo-1414235077428-338989a841e3?w=320&h=200&fit=crop&q=80",
  "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?w=320&h=200&fit=crop&q=80",
  "https://images.unsplash.com/photo-1552566626-52f8b828add9?w=320&h=200&fit=crop&q=80",
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

function thumbSrc(row: Record<string, unknown>): string {
  const u = str(row, "photo_url").trim();
  return u || fallbackPhotoUrl(str(row, "business_id"));
}

const GRAY_NO_PHOTO =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200"><rect width="100%" height="100%" fill="#e2e8f0"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-size="13">No photo</text></svg>'
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

function num(r: Record<string, unknown>, k: string): number | null {
  const v = r[k];
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtPriceTier(tier: unknown): string {
  if (tier == null || tier === "") return "N/A";
  const t = Math.round(Number(tier));
  if (!Number.isFinite(t)) return "N/A";
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
  if (likedIds.value.includes(bid)) return "已标记 👍";
  if (dislikedIds.value.includes(bid)) return "已标记 👎";
  return "无反馈";
}

async function rerunFeedback() {
  if (!lastMode.value) return;
  await run(buildBody(lastMode.value === "discover"));
}

function resetFeedback() {
  likedIds.value = [];
  dislikedIds.value = [];
  void rerunFeedback();
}
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <h2 class="side-title">Search Settings</h2>
      <p class="hint">
        侧栏权重与 d1doc 描述一致；更大 Pool 便于在 👍/👎 后不重搜即可重排。
      </p>

      <label class="chk"
        ><input v-model="forceRebuild" type="checkbox" /> Force Rebuild Index</label
      >

      <h3 class="sub">Ranking weights</h3>
      <label class="sl">w_semantic（文本相似）</label>
      <input v-model.number="wSemantic" type="range" min="0" max="2" step="0.05" />
      <span class="val">{{ wSemantic.toFixed(2) }}</span>

      <label class="sl">w_rating（星级 × review 信任）</label>
      <input v-model.number="wRating" type="range" min="0" max="2" step="0.05" />
      <span class="val">{{ wRating.toFixed(2) }}</span>

      <label class="sl">w_price</label>
      <input v-model.number="wPrice" type="range" min="0" max="2" step="0.05" />
      <span class="val">{{ wPrice.toFixed(2) }}</span>

      <label class="sl">w_distance</label>
      <input v-model.number="wDistance" type="range" min="0" max="2" step="0.05" />
      <span class="val">{{ wDistance.toFixed(2) }}</span>

      <label class="sl">w_popularity（log reviews）</label>
      <input v-model.number="wPopularity" type="range" min="0" max="2" step="0.05" />
      <span class="val">{{ wPopularity.toFixed(2) }}</span>

      <h3 class="sub">Interactive v2 — candidate pool</h3>
      <label class="sl">Internal pool size（15–120）</label>
      <input v-model.number="poolK" type="range" min="15" max="120" step="5" />
      <span class="val">{{ poolK }}</span>
    </aside>

    <section class="main">
      <nav class="back"><router-link to="/">← 首页</router-link></nav>

      <h1>Yelp Commercial &amp; Dining Intelligence</h1>
      <p class="caption">
        排名融合侧栏权重（星级、文本相似、价格、距离、热度）。先
        <strong>Step 1</strong> 选州/市并「泛检索」，再在 <strong>Step 2</strong> 用自然语言与菜系精化。
      </p>

      <div class="card block">
        <h2>Search &amp; filters</h2>
        <p class="hint">
          <strong>Step 1</strong>：必选州，市可选（精确匹配）。首次只按「普通餐厅」排序。<br />
          <strong>Step 2</strong>：菜系、预算、关键词或自然语言 → 点击「Update with preferences」。
        </p>

        <h3>Step 1 — Where are you dining?</h3>
        <div class="row3">
          <div>
            <label>State（必选）</label>
            <select v-model="browseState">
              <option v-for="s in states" :key="s" :value="s">{{ s }}</option>
            </select>
          </div>
          <div>
            <label>City（可选）</label>
            <input v-model="browseCity" type="text" placeholder="精确城市名" />
          </div>
          <div class="btn-cell">
            <button
              type="button"
              class="primary"
              :disabled="loading || !browseState"
              @click="runDiscover"
            >
              Find general restaurants here
            </button>
          </div>
        </div>

        <h3>Step 2 — Refine</h3>
        <details :open="step2Open">
          <summary>Preferences &amp; natural language</summary>
          <label class="mt">Natural language</label>
          <textarea
            v-model="nlQuery"
            rows="3"
            placeholder="cheap sushi, near NYU, within 3 km"
          />

          <label class="mt">Cuisines</label>
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

          <label class="mt">Extra keywords（可选）</label>
          <input v-model="keywords" type="text" />

          <label class="mt">Top-K（展示条数）</label>
          <input v-model.number="topK" type="number" min="3" max="30" class="narrow" />
        </details>

        <button
          type="button"
          class="secondary"
          :disabled="loading || !browseState"
          @click="runRefine"
        >
          Update with preferences
        </button>
      </div>

      <p v-if="err" class="err">{{ err }}</p>

      <div v-if="data && data.results.length" class="card block">
        <h2>Recommendations</h2>
        <p class="hint">{{ metaPool }}</p>
        <details class="mb">
          <summary>Parsed constraints（rule-based）</summary>
          <pre v-if="metaParsed" class="json">{{ JSON.stringify(metaParsed, null, 2) }}</pre>
          <p v-if="resolvedQueryText" class="hint">query_text: <code>{{ resolvedQueryText }}</code></p>
        </details>
        <button type="button" class="ghost" @click="resetFeedback">Reset v2 feedback (likes / dislikes)</button>

        <div
          v-for="(row, i) in data.results"
          :key="str(row as Record<string, unknown>, 'business_id') + String(i)"
          class="result-card"
        >
          <div class="result-inner">
            <div class="thumb-wrap">
              <img
                class="thumb"
                :src="thumbSrc(row as Record<string, unknown>)"
                :alt="str(row as Record<string, unknown>, 'name')"
                loading="lazy"
                referrerpolicy="no-referrer"
                @error="onThumbErr($event, row as Record<string, unknown>)"
              />
            </div>
            <div class="result-body">
              <div class="title-row">
                <div>
                  <strong>#{{ i + 1 }}. {{ str(row as Record<string, unknown>, "name") }}</strong>
                  <p class="subline">
                    {{ str(row as Record<string, unknown>, "address") }},
                    {{ str(row as Record<string, unknown>, "city") }},
                    {{ str(row as Record<string, unknown>, "state") }}
                    · {{ num(row as Record<string, unknown>, "stars")?.toFixed(1) ?? "—" }}★ ({{
                      str(row as Record<string, unknown>, "review_count")
                    }}
                    reviews) · {{ fmtPriceTier((row as Record<string, unknown>).price_tier) }} ·
                    {{ distLabel(row as Record<string, unknown>) }} ·
                    {{ scoreLine(row as Record<string, unknown>) }}
                  </p>
                </div>
              </div>
              <div v-if="str(row as Record<string, unknown>, 'business_id')" class="fb-row">
                <button
                  type="button"
                  class="fb"
                  @click="toggleLike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  👍 Like
                </button>
                <button
                  type="button"
                  class="fb"
                  @click="toggleDislike(str(row as Record<string, unknown>, 'business_id'))"
                >
                  👎 Dislike
                </button>
                <span class="fb-cap">{{
                  feedbackLabel(str(row as Record<string, unknown>, "business_id"))
                }}</span>
              </div>
              <div class="bar">
                <div class="fill" :style="{ width: starBar(row as Record<string, unknown>) * 100 + '%' }" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <p v-else-if="data" class="muted">无结果（请检查州/市数据覆盖或放宽条件）。</p>

      <p class="foot">
        How it works: TF-IDF cosine on aggregated review text, then multi-factor score（侧栏权重）over
        similarity, stars, price vs budget, distance, log reviews — 与后端 recommend_keywords 一致。
      </p>
    </section>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: minmax(260px, 300px) 1fr;
  gap: 1.25rem;
  align-items: start;
  max-width: 1200px;
  margin: 0 auto;
}
@media (max-width: 840px) {
  .layout {
    grid-template-columns: 1fr;
  }
}
.sidebar {
  position: sticky;
  top: 0.5rem;
  padding: 1rem;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.88rem;
}
.side-title {
  margin: 0 0 0.5rem;
  font-size: 1rem;
}
.sub {
  margin: 1rem 0 0.35rem;
  font-size: 0.85rem;
}
.sl {
  display: block;
  margin-top: 0.5rem;
  font-weight: 600;
  font-size: 0.78rem;
}
.val {
  display: inline-block;
  margin-left: 0.35rem;
  color: #475569;
  font-size: 0.8rem;
}
.hint {
  color: #64748b;
  font-size: 0.82rem;
  line-height: 1.45;
}
.chk {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.75rem;
  cursor: pointer;
}
.main {
  min-width: 0;
}
.back {
  margin-bottom: 0.5rem;
}
.back a {
  color: #2563eb;
}
.caption {
  color: #475569;
  font-size: 0.92rem;
  max-width: 52rem;
}
.card.block {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin-top: 1rem;
  background: #fff;
}
.card h2 {
  margin-top: 0;
}
.row3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.75rem;
  margin: 0.75rem 0 1rem;
}
@media (max-width: 720px) {
  .row3 {
    grid-template-columns: 1fr;
  }
}
.row3 label {
  display: block;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 0.2rem;
}
.row3 select,
.row3 input[type="text"] {
  width: 100%;
  padding: 0.35rem 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
}
.btn-cell {
  display: flex;
  align-items: flex-end;
}
button.primary {
  background: #2563eb;
  color: #fff;
  border: none;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}
button.primary:disabled,
button.secondary:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
button.secondary {
  margin-top: 0.75rem;
  background: #e2e8f0;
  border: none;
  padding: 0.45rem 0.85rem;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}
button.ghost {
  margin-bottom: 1rem;
  background: transparent;
  border: 1px dashed #94a3b8;
  padding: 0.35rem 0.65rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
}
.mt {
  display: block;
  margin-top: 0.65rem;
  font-weight: 600;
  font-size: 0.8rem;
}
textarea,
input.narrow {
  width: 100%;
  max-width: 24rem;
  padding: 0.4rem 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
}
.cuisine-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin: 0.35rem 0;
}
.cuisine {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  font-size: 0.85rem;
  cursor: pointer;
}
.err {
  color: #b91c1c;
  margin-top: 0.75rem;
}
.muted {
  color: #64748b;
}
.json {
  background: #f1f5f9;
  padding: 0.75rem;
  font-size: 0.75rem;
  overflow: auto;
  max-height: 14rem;
}
.mb {
  margin-bottom: 0.75rem;
}
.result-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  margin-top: 0.75rem;
  padding: 0.75rem;
  background: #fafafa;
}
.result-inner {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: 0.75rem;
}
@media (max-width: 560px) {
  .result-inner {
    grid-template-columns: 1fr;
  }
}
.thumb-wrap {
  border-radius: 6px;
  min-height: 88px;
  aspect-ratio: 16 / 10;
  max-height: 120px;
  background: #e2e8f0;
  overflow: hidden;
}
.thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.ph {
  background: #e2e8f0;
  border-radius: 6px;
  min-height: 88px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  color: #64748b;
}
.subline {
  margin: 0.25rem 0 0;
  font-size: 0.8rem;
  color: #475569;
}
.fb-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.5rem;
}
.fb {
  padding: 0.25rem 0.55rem;
  font-size: 0.8rem;
  cursor: pointer;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  background: #fff;
}
.fb-cap {
  font-size: 0.75rem;
  color: #64748b;
}
.bar {
  height: 6px;
  background: #e2e8f0;
  border-radius: 4px;
  margin-top: 0.5rem;
  overflow: hidden;
}
.bar .fill {
  height: 100%;
  background: #f59e0b;
}
.foot {
  font-size: 0.8rem;
  color: #64748b;
  margin-top: 1.5rem;
  max-width: 48rem;
}
</style>
