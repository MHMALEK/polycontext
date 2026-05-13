import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Vite dev server proxies FastAPI endpoints so the browser hits one origin
// (avoids CORS preflights on every fetch). In production the React build is
// mounted by FastAPI at /ui.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/ask": "http://127.0.0.1:8000",
      "/decompose": "http://127.0.0.1:8000",
      "/runs": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  base: "./",
});
