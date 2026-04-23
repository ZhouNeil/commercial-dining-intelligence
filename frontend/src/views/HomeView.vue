<script setup lang="ts">
import { onMounted, ref } from "vue";
import { getHealth, type HealthResponse } from "../api/client";

const health = ref<HealthResponse | null>(null);
const err = ref<string | null>(null);

onMounted(async () => {
  try {
    health.value = await getHealth();
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  }
});
</script>

<template>
  <section class="panel">
    <h1>Commercial Dining</h1>
    <p class="muted">先启动后端：<code>./scripts/run_api.sh</code>（端口 8000），再使用下方页面。</p>
    <nav class="nav">
      <router-link to="/search">餐厅检索</router-link>
      <router-link to="/merchant">商家选址预测</router-link>
    </nav>
    <h2>服务健康</h2>
    <p v-if="err" class="err">{{ err }}</p>
    <pre v-else-if="health" class="code">{{ JSON.stringify(health, null, 2) }}</pre>
    <p v-else class="muted">加载中…</p>
  </section>
</template>

<style scoped>
.panel {
  max-width: 48rem;
}
.nav {
  display: flex;
  gap: 1rem;
  margin: 1rem 0;
}
.nav a {
  color: #2563eb;
}
.muted {
  color: #64748b;
  font-size: 0.9rem;
}
.err {
  color: #b91c1c;
}
.code {
  background: #f1f5f9;
  padding: 1rem;
  overflow: auto;
  font-size: 0.8rem;
}
</style>
