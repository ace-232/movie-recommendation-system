import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    build: {
    outDir: "dist",
    emptyOutDir: true
  },
    port: 3000, // The port on which Vite dev server will run
    open: true, // Automatically open the app in the browser
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000", // Flask backend server
        changeOrigin: true, // Modify the origin of the request
        secure: false, // Disable SSL verification for development
        rewrite: (path) => path.replace(/^\/api/, ""), // Remove /api prefix before sending the request to the backend
      },
    },
  },
});
