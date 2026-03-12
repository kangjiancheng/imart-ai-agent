```bash
cd app-ai-simple
uv sync          # creates .venv + installs everything
uv run uvicorn src.main:app --reload --port 9000
```

open: http://127.0.0.1:9000/docs

## Prerequisite

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env
# uv version
# echo 'source $HOME/.local/bin/env' >> ~/.zshrc

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
