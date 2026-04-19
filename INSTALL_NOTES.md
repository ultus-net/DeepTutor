# DeepTutor — Bazzite Native Install Notes

## System Info
- **OS:** Bazzite (Fedora Atomic / immutable)
- **Python:** 3.14.3 (system, via `/usr/bin/python3`)
- **Node.js (system):** v25.9.0 — **NOT compatible** with Next.js 16 SWC binaries
- **Node.js (brew):** v22.22.2 — installed via `brew install node@22`, used for frontend
- **Package manager:** Homebrew (linuxbrew)

## What Was Changed During Installation

### 1. Cloned repo (shallow)
```bash
cd ~/Documents/Learning
git clone --depth 1 https://github.com/HKUDS/DeepTutor.git DeepTutor
```

### 2. Python virtual environment
```bash
cd DeepTutor
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installed Python dependencies
The project's `pyproject.toml` `[server]` extra only covers FastAPI/uvicorn but
**not** the RAG pipeline (LlamaIndex) or LLM provider SDKs. Two steps were needed:

```bash
pip install -e ".[server]"                    # core + server deps
pip install -r requirements/server.txt        # full deps (RAG, LLM SDKs, etc.)
```

**numpy workaround:** `requirements/cli.txt` pins `numpy>=1.24.0,<2.0.0`.
numpy 1.x has no pre-built wheel for Python 3.14 and building from source fails
because Bazzite (atomic) doesn't ship `python3-devel` headers. Fix: install
`numpy` 2.x (which has wheels for 3.14) **before** the requirements file, then
let pip skip the conflicting constraint:

```bash
pip install numpy          # installs 2.x with pre-built wheel
pip install -r requirements/server.txt   # skips numpy since it's satisfied
```

### 4. Installed Node.js 22 via Homebrew
System Node v25.9 causes a **Bus error** when loading `@next/swc-linux-x64-gnu`
(the Next.js SWC native compiler). Node 22 LTS is compatible.

```bash
brew install node@22
```

Binary lives at `/home/linuxbrew/.linuxbrew/opt/node@22/bin/node`.
It is keg-only (not symlinked into PATH by default).

### 5. Installed frontend dependencies with Node 22's npm
```bash
cd web
/home/linuxbrew/.linuxbrew/opt/node@22/bin/npm install
```

### 6. Fixed SWC binary permissions
After `npm install`, the SWC `.node` binary lacked execute permission, causing a
silent Bus error crash. Fix:

```bash
chmod +x node_modules/@next/swc-linux-x64-gnu/next-swc.linux-x64-gnu.node
```

### 7. Copied `.env` configuration
The pre-configured `.env` from the workspace root was copied into the DeepTutor
directory. It uses:
- **LLM:** Claude Sonnet 4.6 via Azure AI Foundry (Anthropic binding)
- **Embedding:** Azure OpenAI text-embedding-3-small
- **Search:** DuckDuckGo (no API key needed)

The frontend also has `web/.env.local` (auto-generated) pointing to the backend:
```
NEXT_PUBLIC_API_BASE=http://localhost:8001
```

## Services

| Service  | Port | URL                      |
|----------|------|--------------------------|
| Backend  | 8001 | http://localhost:8001     |
| Frontend | 3782 | http://localhost:3782     |
