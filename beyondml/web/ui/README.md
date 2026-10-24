# BeyondML web UI

React + Vite + TypeScript + Tailwind + Plotly. Six screens over the gateway's REST and SSE routes
(`beyondml/gateway/app.py`): plan board, agent (instruction / live / outputs / evolve), chat dock, approvals,
experiments, files.

```bash
npm install
npm run dev      # Vite on :5173, proxying /projects to the gateway on :7777 (`beyondml serve`)
npm run build    # tsc -b && vite build → dist/, which the gateway serves at /
```

`src/session.ts` folds the event stream the same way `beyondml/gateway/state.py` does; keep them in step.
