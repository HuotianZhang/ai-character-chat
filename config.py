"""
AI Character System - Global Configuration
"""
import os

# === LLM API Configuration ===
# Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "Your API")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Generation parameters
LLM_TEMPERATURE = 0.85  # Creativity level for character dialogue
LLM_MAX_TOKENS = 4096   # Must be large: Gemini 2.5 Flash thinking tokens count toward this limit

# === Storage Configuration ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHARACTER_FILE = os.path.join(DATA_DIR, "character.json")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
STORYLINE_FILE = os.path.join(DATA_DIR, "storyline.json")

# === System Parameters ===
# Memory system
MAX_SHORT_TERM_MEMORY = 30      # Short-term memory keeps the last N conversations
MAX_LONG_TERM_MEMORY = 100      # Long-term memory maximum entries
MEMORY_CONSOLIDATION_THRESHOLD = 0.6  # Memories exceeding this emotion intensity are consolidated

# Emotion system
MOOD_DECAY_INTERVAL_MINUTES = 30    # Emotion decay check interval
MOOD_BASELINE = 6.0                 # Emotion baseline (1-10)

# Affinity system
AFFINITY_INITIAL = 65               # Initial affinity
AFFINITY_MIN = 0
AFFINITY_MAX = 100
SPECIAL_AFFINITY_INITIAL = 65       # Special affinity initial value

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
