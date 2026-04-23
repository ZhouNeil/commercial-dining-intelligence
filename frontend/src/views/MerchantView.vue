<script setup lang="ts">
import { computed, ref } from "vue";
import { postMerchantPredict, type MerchantPredictResponse } from "../api/client";

const city = ref("Philadelphia");
const lat = ref(39.9526);
const lon = ref(-75.1652);
/** 逗号或换行分隔的 cat_* 列名 */
const categoriesText = ref("cat_coffee_&_tea, cat_fast_food");
const maxRows = ref(2000);
const loading = ref(false);
const err = ref<string | null>(null);
const result = ref<MerchantPredictResponse | null>(null);

const categoryKeys = computed(() =>
  categoriesText.value
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
);

async function run() {
  err.value = null;
  result.value = null;
  const keys = categoryKeys.value;
  if (!keys.length) {
    err.value = "请至少填写一个品类列名（如 cat_fast_food）。";
    return;
  }
  loading.value = true;
  try {
    result.value = await postMerchantPredict({
      city: city.value.trim() || null,
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
</script>

<template>
  <section class="panel">
    <nav class="back"><router-link to="/">← 首页</router-link></nav>
    <h1>商家选址预测</h1>
    <p class="muted">
      调用 <code>POST /api/v1/merchant/predict</code>。需 <code>data/train_spatial.csv</code> 与两个
      <code>global_*_model.pkl</code>。
    </p>

    <div class="form">
      <label>城市（过滤参考商户，可空）</label>
      <input v-model="city" type="text" />

      <label>纬度 / 经度</label>
      <div class="row">
        <input v-model.number="lat" type="number" step="any" />
        <input v-model.number="lon" type="number" step="any" />
      </div>

      <label>品类列名（<code>cat_*</code>，逗号或换行分隔）</label>
      <textarea v-model="categoriesText" rows="4" placeholder="cat_coffee_&_tea&#10;cat_fast_food" />

      <label>无 city 列时参考行数上限</label>
      <input v-model.number="maxRows" type="number" min="100" max="50000" class="short" />

      <button type="button" :disabled="loading" @click="run">
        {{ loading ? "计算中…" : "计算预测" }}
      </button>
    </div>

    <p v-if="err" class="err">{{ err }}</p>

    <div v-if="result" class="cards">
      <div class="card">
        <div class="label">开业概率</div>
        <div class="value">{{ (result.survival_probability * 100).toFixed(1) }}%</div>
      </div>
      <div class="card">
        <div class="label">预测星级</div>
        <div class="value">{{ result.predicted_stars.toFixed(2) }} / 5</div>
      </div>
      <div class="card">
        <div class="label">参考商户数</div>
        <div class="value">{{ result.reference_row_count }}</div>
      </div>
    </div>

    <template v-if="result && Object.keys(result.live_feature_preview || {}).length">
      <h2>特征预览</h2>
      <pre class="code">{{ JSON.stringify(result.live_feature_preview, null, 2) }}</pre>
    </template>
  </section>
</template>

<style scoped>
.panel {
  max-width: 40rem;
}
.back {
  margin-bottom: 0.5rem;
}
.back a {
  color: #2563eb;
}
.muted {
  color: #64748b;
  font-size: 0.9rem;
}
.err {
  color: #b91c1c;
  margin-top: 1rem;
}
.form {
  display: grid;
  gap: 0.5rem;
  margin-top: 1rem;
}
.form label {
  font-weight: 600;
  font-size: 0.85rem;
}
.form input,
.form textarea {
  padding: 0.4rem 0.5rem;
  border: 1px solid #cbd5e1;
  border-radius: 4px;
  font-family: inherit;
}
.form input.short {
  max-width: 8rem;
}
.row {
  display: flex;
  gap: 0.5rem;
}
.row input {
  flex: 1;
}
.form button {
  margin-top: 0.5rem;
  padding: 0.5rem 1rem;
  cursor: pointer;
}
.cards {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1.5rem;
}
.card {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 1rem 1.25rem;
  min-width: 8rem;
}
.card .label {
  font-size: 0.75rem;
  color: #64748b;
}
.card .value {
  font-size: 1.35rem;
  font-weight: 700;
  margin-top: 0.25rem;
}
.code {
  background: #f1f5f9;
  padding: 1rem;
  font-size: 0.8rem;
  overflow: auto;
}
</style>
