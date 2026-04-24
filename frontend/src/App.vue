<script setup lang="ts">
</script>

<template>
  <div class="shell">
    <header class="top">
      <router-link to="/" class="brand">
        <span class="brand-mark" aria-hidden="true" />
        <span class="brand-text">Dining<span class="brand-accent">Intel</span></span>
      </router-link>
      <nav class="nav">
        <router-link to="/">Home</router-link>
        <router-link to="/search">Discover</router-link>
        <router-link to="/merchant">Site predictor</router-link>
      </nav>
    </header>
    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<style>
:root {
  --font: "Plus Jakarta Sans", system-ui, -apple-system, sans-serif;
  --ink: #1c1917;
  --ink-muted: #57534e;
  --line: #e7e5e4;
  --surface: #fafaf9;
  --surface-elevated: #ffffff;
  --accent: #dc2626;
  --accent-soft: #fef2f2;
  --accent-hover: #b91c1c;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: var(--font);
  color: var(--ink);
  background: var(--surface);
  -webkit-font-smoothing: antialiased;
}

.shell {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.65rem 1.5rem;
  background: var(--surface-elevated);
  border-bottom: 1px solid var(--line);
  position: sticky;
  top: 0;
  z-index: 50;
  box-shadow: 0 1px 0 rgba(28, 25, 23, 0.04);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  text-decoration: none;
  color: var(--ink);
  font-weight: 800;
  font-size: 1.05rem;
  letter-spacing: -0.03em;
}

.brand-mark {
  width: 2rem;
  height: 2rem;
  border-radius: 10px;
  background: linear-gradient(135deg, #b91c1c 0%, #dc2626 50%, #ef4444 100%);
  box-shadow: 0 2px 8px rgba(220, 38, 38, 0.35);
}

.brand-accent {
  color: var(--accent);
  font-weight: 800;
}

.nav {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.nav a {
  color: var(--ink-muted);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 600;
  padding: 0.45rem 0.75rem;
  border-radius: 999px;
  transition: color 0.15s, background 0.15s;
}

.nav a:hover {
  color: var(--ink);
  background: #f5f5f4;
}

.nav a.router-link-active {
  color: var(--accent);
  background: var(--accent-soft);
}

.main {
  flex: 1;
  padding: 0;
}
</style>
