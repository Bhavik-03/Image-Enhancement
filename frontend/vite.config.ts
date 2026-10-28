import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // the FastAPI backend runs separately in dev (`uvicorn api:app --reload`)
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
