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
  <div class="home">
    <section class="hero">
      <p class="eyebrow">Commercial dining intelligence</p>
      <h1>Pick a table worth the trip</h1>
      <p class="lede">
        Semantic search over restaurant reviews with multi-signal ranking and personalized re-ranking.
        Discover great spots as a tourist or predict site viability as a merchant.
      </p>
      <div class="cta-row">
        <router-link class="btn btn-primary" to="/search">Discover restaurants</router-link>
        <router-link class="btn btn-secondary" to="/merchant">Site predictor</router-link>
      </div>
    </section>

    <section class="cards">
      <router-link to="/search" class="feature-card">
        <div class="fc-icon" aria-hidden="true">◇</div>
        <h2>Discover</h2>
        <p>State, city, cuisines, natural language, and live re-ranking from likes and dislikes.</p>
        <span class="fc-link">Open search →</span>
      </router-link>
      <router-link to="/merchant" class="feature-card">
        <div class="fc-icon fc-icon--alt" aria-hidden="true">◎</div>
        <h2>Site predictor</h2>
        <p>Survival-style probability and predicted stars from spatial features and category mix.</p>
        <span class="fc-link">Open tool →</span>
      </router-link>
    </section>

    <section class="health-panel">
      <h2>API health</h2>
      <p v-if="err" class="err">{{ err }}</p>
      <pre v-else-if="health" class="code">{{ JSON.stringify(health, null, 2) }}</pre>
      <p v-else class="muted">Loading…</p>
    </section>
  </div>
</template>

<style scoped>
.home {
  max-width: 56rem;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 3.5rem;
}

.hero {
  margin-bottom: 2.5rem;
}

.eyebrow {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #a8a29e;
  margin: 0 0 0.75rem;
}

.hero h1 {
  margin: 0 0 1rem;
  font-size: clamp(1.85rem, 4vw, 2.35rem);
  font-weight: 800;
  letter-spacing: -0.035em;
  line-height: 1.15;
  color: #1c1917;
}

.lede {
  margin: 0 0 1.5rem;
  color: #57534e;
  font-size: 1.05rem;
  line-height: 1.6;
  max-width: 40rem;
}

.lede code {
  font-size: 0.85em;
  background: #f5f5f4;
  padding: 0.12em 0.35em;
  border-radius: 6px;
  color: #44403c;
}

.cta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.65rem 1.25rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.9rem;
  text-decoration: none;
  transition: transform 0.15s, box-shadow 0.15s;
}

.btn-primary {
  background: linear-gradient(135deg, #b91c1c, #ef4444);
  color: #fff;
  box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35);
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(185, 28, 28, 0.4);
}

.btn-secondary {
  background: #fff;
  color: #1c1917;
  border: 1px solid #e7e5e4;
}

.btn-secondary:hover {
  border-color: #d6d3d1;
  background: #fafaf9;
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.25rem;
  margin-bottom: 2.5rem;
}

.feature-card {
  display: block;
  padding: 1.5rem 1.35rem;
  background: #fff;
  border: 1px solid #e7e5e4;
  border-radius: 20px;
  text-decoration: none;
  color: inherit;
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
  box-shadow: 0 4px 24px rgba(28, 25, 23, 0.06);
}

.feature-card:hover {
  border-color: #fca5a5;
  box-shadow: 0 12px 40px rgba(220, 38, 38, 0.12);
  transform: translateY(-2px);
}

.fc-icon {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 12px;
  background: #fef2f2;
  color: #dc2626;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  margin-bottom: 1rem;
}

.fc-icon--alt {
  background: #ffe4e6;
  color: #be123c;
}

.feature-card h2 {
  margin: 0 0 0.5rem;
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.feature-card p {
  margin: 0 0 1rem;
  font-size: 0.9rem;
  color: #78716c;
  line-height: 1.55;
}

.fc-link {
  font-size: 0.82rem;
  font-weight: 700;
  color: #dc2626;
}

.health-panel h2 {
  font-size: 0.95rem;
  font-weight: 800;
  margin: 0 0 0.75rem;
  color: #44403c;
}

.muted {
  color: #a8a29e;
  font-size: 0.95rem;
}

.err {
  color: #b91c1c;
  font-size: 0.9rem;
}

.code {
  background: #1c1917;
  color: #e7e5e4;
  padding: 1rem 1.1rem;
  border-radius: 14px;
  overflow: auto;
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0;
}
</style>
