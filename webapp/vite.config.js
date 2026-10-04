import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Served from https://julianschelb.github.io/retexo/demo/, next to the documentation
export default defineConfig({
  base: "/retexo/demo/",
  plugins: [react(), tailwindcss()],
});
