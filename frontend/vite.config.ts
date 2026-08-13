import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));

// GHPAGES_BASE=/gpu-optimizer/ is set by the Pages deploy workflow so asset
// URLs resolve under the project-pages subpath.
export default defineConfig({
  base: process.env.GHPAGES_BASE ?? "/",
  define: { __APP_VERSION__: JSON.stringify(pkg.version) },
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
