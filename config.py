from pathlib import Path
import os

APP_NAME = "DataMind AI"
APP_SUBTITLE = "Intelligent Statistical Analysis & Machine Learning Agent"
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
SAMPLE_DIR = ROOT / "sample_data"
MAX_UPLOAD_MB = int(os.getenv("DATAMIND_MAX_UPLOAD_MB", "200"))
DEFAULT_RANDOM_STATE = 42

NAVY = "#0B1F3A"
TEAL = "#0F766E"
SLATE = "#475569"
LIGHT = "#F8FAFC"

AI_PROVIDER = os.getenv("DATAMIND_AI_PROVIDER", "openai_compatible")
AI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
AI_MODEL = os.getenv("DATAMIND_AI_MODEL", "gpt-4o-mini")
