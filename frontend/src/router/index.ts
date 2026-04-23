import { createRouter, createWebHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import SearchView from "../views/SearchView.vue";
import MerchantView from "../views/MerchantView.vue";

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: "/", name: "home", component: HomeView },
    { path: "/search", name: "search", component: SearchView },
    { path: "/merchant", name: "merchant", component: MerchantView },
  ],
});
