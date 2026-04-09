# DeepTutor — Local Docker Setup (Ollama / No API Keys)

### Prerequisites
- Docker ([docs.docker.com/engine/install](https://docs.docker.com/engine/install/))
- ~12 GB free disk space (models + image)

---

### 1. Clone your fork
```bash
git clone https://github.com/ultus-net/DeepTutor.git
cd DeepTutor
git checkout local-ollama-setup
```

### 2. Create your `.env`
```bash
cp .env.example .env
```
Then edit `.env` — the minimum required for the local Ollama setup:
```ini
LLM_BINDING=openai
LLM_MODEL=gemma4:e4b
LLM_API_KEY=ollama
LLM_HOST=http://ollama:11434/v1

EMBEDDING_BINDING=openai
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_API_KEY=ollama
EMBEDDING_HOST=http://ollama:11434/v1
EMBEDDING_DIMENSION=768
```

### 3. Pull the images
```bash
docker compose -f docker-compose.local.yml pull
```

### 4. Start Ollama and download models
```bash
docker compose -f docker-compose.local.yml up -d ollama

# Wait ~10s for Ollama to be healthy, then pull models (~10 GB total)
docker compose -f docker-compose.local.yml exec ollama ollama pull gemma4:e4b
docker compose -f docker-compose.local.yml exec ollama ollama pull nomic-embed-text:latest
```

### 5. Start DeepTutor
```bash
docker compose -f docker-compose.local.yml up -d deeptutor
```

Open **http://localhost:3782**

---

### Useful commands
```bash
# Tail logs
docker compose -f docker-compose.local.yml logs -f

# Stop everything
docker compose -f docker-compose.local.yml down

# Restart after a reboot (models are persisted in the ollama_data volume)
docker compose -f docker-compose.local.yml up -d
```

### Model swap
To use a different model, edit `.env` and restart:
```bash
# e.g. switch to gemma4:26b (needs ~18 GB RAM)
docker compose -f docker-compose.local.yml exec ollama ollama pull gemma4:26b
# update LLM_MODEL=gemma4:26b in .env, then:
docker compose -f docker-compose.local.yml up -d deeptutor
```

> **Note:** On first `up -d` after a reboot, Ollama takes ~15s to become healthy before DeepTutor starts. Models are stored in the `ollama_data` Docker volume and do **not** need to be re-downloaded.
