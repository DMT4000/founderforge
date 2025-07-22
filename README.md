# FounderForge AI Cofounder

A localhost-first AI virtual cofounder system designed to assist entrepreneurs with strategic decision-making, funding guidance, and operational support.

## Quick Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Initialize Database** (will be created automatically on first run):
   ```bash
   python -c "from config.settings import settings; print(f'Database will be created at: {settings.database_path}')"
   ```

## Project Structure

```
FounderForge/
├── src/                    # Core application code
├── tests/                  # Test suite
├── data/                   # Local data storage (SQLite, FAISS, logs)
├── config/                 # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## Key Features

- **Localhost-First**: All data stays local except Gemini API calls
- **Multi-Agent System**: LangGraph-powered agent workflows
- **Persistent Memory**: SQLite + FAISS for context retention
- **Feature Flags**: JSON-based configuration for rapid iteration
- **Git Integration**: Prompt versioning and experiment tracking

## Dependencies

- **langgraph**: Multi-agent workflow orchestration
- **google-generativeai**: Gemini 2.5 Flash API integration
- **faiss-cpu**: Vector search for semantic memory
- **sqlite3**: Local database storage
- **streamlit**: Web interface
- **sentence-transformers**: Local text embeddings
- **pytest**: Testing framework
- **psutil**: System monitoring

## Configuration

The system uses a combination of:
- Environment variables (`.env` file)
- Feature flags (`config/feature_flags.json`)
- Git-based prompt versioning

See `config/settings.py` for all configuration options.