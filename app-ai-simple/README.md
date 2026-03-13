## Development

**First time only:**

```bash
cd app-ai-simple
uv sync          # creates .venv + installs everything
```

**Daily (just run this):**

```bash
uv run uvicorn src.main:app --reload --port 9000
```

open: http://127.0.0.1:9000/docs

## Prerequisite

### Python Manager uv

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv version

# Create venv + install all deps from pyproject.toml
uv sync

# Add a new package
uv add fastapi

# Remove a package
uv remove openai

# Run a command inside the venv
uv run uvicorn src.main:app --reload

# Install a specific Python version
uv python install 3.12

```

### Milvus setup

set `.env` first

```ini
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

**Option B Docker Compose**

https://milvus.io/docs/install_standalone-docker-compose.md

```bash
 curl -sfL https://github.com/milvus-io/milvus/releases/download/v2.6.11/milvus-standalone-docker-compose.yml -O docker-compose.yml

 # Terminal 1 — start Milvus stack (etcd + minio + milvus)
docker compose -f milvus-standalone-docker-compose.yml up -d

# Terminal 2 — run app-ai with hot reload
cd app-ai
source venv/bin/activate
uvicorn src.main:app --reload --port 8000
```

**Option A: Docker Milvus via `standalone_embed.sh`**

The included `standalone_embed.sh` script manages a Milvus standalone Docker container (version `v2.6.12`). It creates `embedEtcd.yaml` and `user.yaml` config files automatically, and persists data in `./volumes/milvus/`.

```bash
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# First run — pulls the Milvus image and starts the container (sudo required)
bash standalone_embed.sh start
# → waits until container is healthy (~30s)
```

more base:

```bash
# Check container is running
docker ps | grep milvus-standalone

# Stop (data is preserved in ./volumes/milvus/)
bash standalone_embed.sh stop

# Restart
bash standalone_embed.sh restart

# Delete container + all data (prompts for confirmation — irreversible)
bash standalone_embed.sh delete

# Upgrade to latest Milvus version
bash standalone_embed.sh upgrade
```
