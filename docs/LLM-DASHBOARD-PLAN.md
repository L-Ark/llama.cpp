# LLM Management Dashboard — Project Plan

## Overview
A standalone SvelteKit web application that serves as a management interface for llama-server's router mode API. Hosted separately from llama-server on port 3000, providing model management, HuggingFace model discovery/download, MCP server for Copilot CLI/SDK, and user authentication.

## Architecture
```
Browser → Dashboard (:3000) → llama-server Router API (:8080)
                            → HuggingFace API (model search/download)
                            → MCP Server (:3001) → Copilot CLI/SDK
```

## Tech Stack
- **Frontend**: SvelteKit 5 (matches llama.cpp WebUI ecosystem)
- **Backend**: SvelteKit server routes (API proxy + business logic)
- **Database**: SQLite (via better-sqlite3) for user accounts, download history, presets
- **Styling**: Tailwind CSS
- **Runtime**: Node.js 20+

## Pages & Features

### 1. Dashboard (Home)
- System overview: GPU status, VRAM usage per GPU, loaded models
- Quick actions: load/unload models, open chat
- Server health check (llama-server connectivity)

### 2. Models Manager
- List all discovered models with status (loaded/unloaded/loading)
- Load/unload buttons with VRAM impact preview
- Per-model config: context size, KV cache type, GPU layers
- Model info: file size, quant type, parameter count, architecture
- API: Calls GET /v1/models, POST /models/load, POST /models/unload

### 3. Download Center
- Search HuggingFace for GGUF models (uses HF API)
- Filter by: model family, quant type, size, popularity
- Show available quants with size estimates
- One-click download to /mnt/models/gguf with progress bar
- Download queue (multiple concurrent downloads)
- Backend: Python helper script or Node fetch → HF API

### 4. Chat Playground
- Option A: Enhanced chat UI (SvelteKit) talking to /v1/chat/completions
- Option B: Embed/link to llama-server's built-in WebUI at :8080
- Model selector dropdown (from /v1/models)
- Conversation history (stored in IndexedDB or SQLite)

### 5. Settings & Admin
- llama-server connection config (host, port)
- Default GPU config (tensor-split, split-mode, ngl)
- Default inference config (context size, KV cache type, parallel)
- User management (basic auth: username/password)
- API key management for external access

### 6. MCP Server (Copilot Integration)
- Express/Fastify sidecar or SvelteKit API route
- Exposes llama-server as MCP-compatible endpoint
- Registration config for Copilot CLI (~/.copilot/mcp.json)
- Connection status & request logging
- Model selection for Copilot requests

## Milestones

### M1: Foundation
- SvelteKit project scaffolding + Tailwind
- llama-server API client (TypeScript)
- Dashboard page with GPU/model status
- Models page with load/unload

### M2: Download Center
- HuggingFace API integration
- GGUF model search & filtering
- Download manager with progress tracking

### M3: Chat & Settings
- Chat playground with model selection
- Settings page with config persistence
- Basic auth (username/password)

### M4: MCP Server
- MCP protocol implementation
- Copilot CLI registration helper
- Request routing to llama-server

### M5: Polish & Open Source
- Error handling, loading states, notifications
- Mobile-responsive layout
- Documentation & README
- GitHub repo setup, CI/CD

## llama-server API Reference (Router Mode)
```
GET  /v1/models                    # List all models with status
POST /v1/chat/completions          # Chat (auto-loads model)
POST /v1/completions               # Text completion
POST /v1/embeddings                # Embeddings
POST /models/load   {model: "..."}  # Manually load a model
POST /models/unload {model: "..."}  # Unload a model
GET  /health                        # Server health check
GET  /metrics                       # Prometheus metrics
```

## Deployment
- Runs inside LXC 200 alongside llama-server
- Node.js process on port 3000
- systemd service for auto-start
- Accessible at http://192.168.0.212:3000

### M6: Long-Term Memory
- Persistent conversation storage (SQLite) with search
- Cross-device sync (conversations stored server-side, not just browser)
- User profile & preferences (system prompts, persona presets per user)
- Memory injection: summarize past conversations and inject into system prompt
- RAG-style fact store: extract user facts/preferences, retrieve relevant ones per query
- Copilot session memory: MCP context carries prior session summaries

## Future Considerations
- TurboQuant KV cache support (when ported to upstream)
- Model fine-tuning integration (LoRA)
- Benchmark runner (automated perf testing)
- Multi-server support (multiple llama-server instances)
- OpenAI API proxy (drop-in replacement for cloud APIs)
