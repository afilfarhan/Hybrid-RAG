"""
Hybrid RAG - Clean Architecture

Project Structure:
- app/           - Main entry point and FastAPI app
- services/      - Core services (embedding, vector store, retrieval, generation)
- api/           - REST API endpoints
- ui/            - Frontend UI (static files)
- data/          - Sample documents and data
- config.py      - Configuration management
"""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent

# Directory structure
APP_DIR = ROOT_DIR / "app"
SERVICES_DIR = ROOT_DIR / "services"
API_DIR = ROOT_DIR / "api"
UI_DIR = ROOT_DIR / "ui"
DATA_DIR = ROOT_DIR / "data"
CONFIG_FILE = ROOT_DIR / ".env"

print(f"Project Structure:")
print(f"  Root: {ROOT_DIR}")
print(f"  App: {APP_DIR}")
print(f"  Services: {SERVICES_DIR}")
print(f"  API: {API_DIR}")
print(f"  UI: {UI_DIR}")
print(f"  Data: {DATA_DIR}")
