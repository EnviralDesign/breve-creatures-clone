# Breve Creatures Clone

![Project Image](images/image.png)

Browser-based evolutionary creature simulation shipped as a single Rust executable.

## Architecture

- One process, one runtime: the Rust binary serves simulation APIs and the browser UI.
- Evolution runs continuously in Rust; browser sessions are control/view clients and can disconnect/reconnect without resetting progress.
- UI assets in `ui/` are embedded at compile-time and served from memory.
- There is no separate frontend server/process to run.

## Repo Layout

- `Cargo.toml`, `Cargo.lock` - executable crate definition
- `src/` - simulation engine, API endpoints, and static asset serving
- `ui/` - browser UI assets bundled into the executable
- `docs/` - protocol and architecture notes

## Quick Start

Prereqs:

- Rust toolchain (`rustup`, `cargo`)

1. Start app

```powershell
cargo run --release
```

2. Open app

- `http://127.0.0.1:8787`
- If port `8787` is already in use, open the fallback URL printed in startup logs.

## Runtime Config

- `SIM_PORT`: override bind port (default `8787`)
- `SIM_MAX_CONCURRENT_JOBS`: cap simulation worker concurrency (default `available_cores - 1`, minimum `1`)

## Persistence

- Runtime checkpoints are stored in `data/checkpoints/`.
- `latest.json` is updated on manual save and autosave.
- Autosave runs every 5 completed generations.
- The server attempts to load `data/checkpoints/latest.json` on startup if present.

## Distributed Computing (Satellite Mode)

The simulation supports distributed computing across your local area network (LAN) or localhost. You can run one main "Primary" node that coordinates evolution, and multiple headless "Satellite" nodes that only run physics simulations, sending results back to the primary.

1. **Start the Primary Node:**
The primary automatically binds to `0.0.0.0` making it ready to accept satellite connections immediately.
```powershell
cargo run --release
```

2. **Start Satellite Node(s):**
On another machine (or the same machine), start the app with the `--satellite` flag pointing to the primary node's URL. Satellites run in headless mode without the UI.
```powershell
cargo run --release -- --satellite ws://<PRIMARY_IP>:8787
```

- Satellites automatically discover your core count and maximize throughput.
- You can add or remove satellites at any time during a simulation without breaking progress.
- Disconnected satellites automatically retry connecting to the primary node.

## Endpoints

- `GET /`
- `GET /health`
- `GET /api/evolution/state`
- `GET /api/evolution/history`
- `POST /api/evolution/control`
- `GET /api/evolution/ws`
- `GET /api/evolution/genome/current`
- `GET /api/evolution/genome/best`
- `POST /api/evolution/genome/import`
- `POST /api/evolution/checkpoint/save`
- `GET /api/evolution/checkpoint/list`
- `POST /api/evolution/checkpoint/load`
- `GET /api/trial/ws`
- `POST /api/eval/generation`
- `GET /api/eval/ws`
- `GET /api/satellite/ws`

Detailed request/stream payload contracts: `docs/api.md`.
