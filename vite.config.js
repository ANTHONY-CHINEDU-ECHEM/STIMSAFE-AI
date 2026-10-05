import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the dashboard runs on port 5173 and proxies API calls to FastAPI on port 8000.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
