# Breve Creatures Clone

![Project Image](images/image.png)

Browser-based evolutionary creature simulation shipped as a single Rust executable.

## Architecture

- One process, one runtime: the Rust binary serves simulation APIs and the browser UI.
- UI assets in `ui/` are embedded at compile-time and served from memory.
- There is no separate frontend server/process to run.

## Repo Layout

- `Cargo.toml`, `Cargo.lock` - executable crate definition
- `src/` - simulation engine, API endpoints, and static asset serving
- `ui/` - browser UI assets bundled into the executable
- `docs/` - protocol and architecture notes
- `tools/` - local helper tooling

## Quick Start

Prereqs:

- Rust toolchain (`rustup`, `cargo`)

1. Start app

```powershell
cargo run
```

2. Open app

- `http://127.0.0.1:8787`

## Runtime Config

- `SIM_PORT`: override bind port (default `8787`)
- `SIM_MAX_CONCURRENT_JOBS`: cap simulation worker concurrency (default `available_cores - 1`, minimum `1`)

## Endpoints

- `GET /`
- `GET /health`
- `GET /api/trial/ws`
- `POST /api/eval/generation`
- `GET /api/eval/ws`

Detailed request/stream payload contracts: `docs/api.md`.
