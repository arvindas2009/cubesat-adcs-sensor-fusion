import { defineConfig } from 'vitest/config';

// Frontend of the browser simulator. FastAPI serves dist/ at "/" (docs/specs/webapp_protocol.md §4);
// during development the Vite server proxies /api (REST and WebSocket) to the Python server.
export default defineConfig({
  root: '.', // webapp/frontend: npm scripts always run from the package directory
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 800, // one bundle: three.js alone is ~650 kB minified (a local tool, no code splitting needed)
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
