# Frontend

Browser UI for the creature simulation, served by the Rust backend.

## What Is Here

- `index.html`: simulation UI and client logic

## Run

From the `backend/` directory:

```powershell
cd backend
cargo run
```

Frontend URL: `http://127.0.0.1:8787`

Health check: `http://127.0.0.1:8787/health`

## Notes

- Frontend API/WebSocket URLs are same-origin and are derived from `window.location`.
- `index.html` is embedded at compile-time into the backend binary.
