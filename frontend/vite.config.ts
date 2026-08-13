import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// GHPAGES_BASE=/gpu-optimizer/ is set by the Pages deploy workflow so asset
// URLs resolve under the project-pages subpath.
export default defineConfig({
  base: process.env.GHPAGES_BASE ?? "/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
