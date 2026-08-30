import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8765",
      "/orders": "http://127.0.0.1:8765",
      "/order": "http://127.0.0.1:8765",
      "/demo": "http://127.0.0.1:8765",
      "/explain": "http://127.0.0.1:8765",
    },
  },
});
