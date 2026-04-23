<script setup lang="ts">
</script>

<template>
  <div class="app">
    <header class="top">
      <router-link to="/" class="brand">Commercial Dining</router-link>
      <nav>
        <router-link to="/search">检索</router-link>
        <router-link to="/merchant">商家选址</router-link>
      </nav>
    </header>
    <main class="main">
      <router-view />
    </main>
  </div>
</template>

<style>
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
  color: #0f172a;
}
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1.25rem;
  background: #0f172a;
  color: #f8fafc;
}
.top a {
  color: #e2e8f0;
  text-decoration: none;
  margin-left: 1rem;
  font-size: 0.9rem;
}
.top a.router-link-active {
  color: #fff;
  font-weight: 600;
}
.brand {
  font-weight: 700;
  color: #fff !important;
  margin-left: 0 !important;
}
.main {
  flex: 1;
  padding: 1.25rem;
}
</style>
