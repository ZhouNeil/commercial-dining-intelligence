#!/bin/bash

# Deactivate conda (if active)
conda deactivate 2>/dev/null || true

# Deactivate uv (if active)
deactivate 2>/dev/null || true

# Install uv (if not installed). WARNING: the next line pipes a script from the web to sh.
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate 2>/dev/null || true

# Install our project requirements
uv pip install -r requirements.txt