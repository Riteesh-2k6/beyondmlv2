import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the gateway runs on 7777 (`beyondml serve`); Vite proxies its API routes.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/projects': { target: 'http://127.0.0.1:7777', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 1400, // plotly-basic is lazy-loaded into its own chunk
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('plotly')) return 'plotly'
        },
      },
    },
  },
})
