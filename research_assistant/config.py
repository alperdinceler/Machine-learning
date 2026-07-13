import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_SEARCH_RESULTS = 5
REQUEST_TIMEOUT = 15  # seconds

RATE_LIMIT_DELAY = 4.0  # seconds between successive API calls
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0

USE_LOCAL_FALLBACK = True
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # default to qwen2.5:7b, can be overridden in .env

