import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Vite dev server proxies FastAPI endpoints so the browser hits one origin
// (avoids CORS preflights on every fetch). In production the React build is
// mounted by FastAPI at /ui.
//
// Ports come from the env so they stay in lockstep with the Makefile and
// docker-compose. Defaults match .env.example.
const API_PORT = Number(process.env.API_PORT) || 18000;
const UI_PORT = Number(process.env.UI_PORT) || 15173;
const apiTarget = `http://127.0.0.1:${API_PORT}`;

export default defineConfig(({ command }) => ({
  plugins: [react(), tailwindcss()],
  server: {
    port: UI_PORT,
    proxy: {
      "/health": apiTarget,
      "/v1": apiTarget,
      "/runs": apiTarget,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  // Absolute base when bundled into FastAPI at /ui/ so `/ui` and `/ui/` both load JS/CSS.
  // Dev server keeps `/` root.
  base: command === "build" ? "/ui/" : "/",
}));
