# Deep Restore — frontend

React + TypeScript + Tailwind v4 UI for the Image Enhancement API. Drag in a
photo or video, pick a model, get a before/after comparison (a draggable
slider for photos, synced side-by-side players for video) and a download.

## Develop

Run the backend and frontend as two processes; Vite proxies `/api/*` to it
(see `vite.config.ts`):

```
# terminal 1, from the repo root
source .venv/bin/activate && uvicorn api:app --reload

# terminal 2, from frontend/
npm install
npm run dev        # http://localhost:5173
```

## Build

```
npm run build       # writes dist/
```

`api.py` serves `frontend/dist/` itself when it exists, so after building,
`python api.py` alone (from the repo root) runs the whole app on one port —
no separate frontend server needed.

## Layout

- `src/lib/api.ts` — fetch wrappers for the backend (`/api/models`,
  `/api/enhance/photo`, the video job endpoints).
- `src/components/UploadZone.tsx` — drag-and-drop + file picker, detects
  photo vs. video by extension/MIME type.
- `src/components/CompareSlider.tsx` — draggable before/after for photos.
- `src/components/VideoCompare.tsx` — synced side-by-side video players.
- `src/components/ProgressRing.tsx` — the circular progress indicator shown
  while a video job runs (driven by `enhance_video`'s `on_frame` callback).
- `src/App.tsx` — ties it together: upload → model pick → enhance → result.
