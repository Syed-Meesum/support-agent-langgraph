import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API base URL is empty by default, so every request is same-origin:
//   - `npm run dev`   -> the proxy below forwards /api to the local backend
//   - Docker/nginx    -> nginx reverse-proxies /api to the backend container
// That means no CORS and no build-time URL to get wrong. Set VITE_API_URL only
// if you are serving this bundle somewhere the backend is not reachable at /api.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
