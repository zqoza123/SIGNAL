"""
Configuration for Political Conviction Signal Engine.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
CONGRESS_API_KEY = os.getenv("CONGRESS_API_KEY", "")
CONGRESS_API_BASE = "https://api.congress.gov/v3"

# --- Baseline Settings ---
BASELINE_WINDOW_DAYS = 90          # Rolling window for baseline calculation
MIN_SAMPLES_FOR_BASELINE = 30     # Minimum speeches before baseline is valid
MIN_WORD_COUNT = 100               # Minimum words in a speech to analyze

# --- Anomaly Detection Settings ---
ZSCORE_THRESHOLD_STRONG = 2.0     # Strong signal threshold (σ)
ZSCORE_THRESHOLD_MODERATE = 1.5   # Moderate signal threshold (σ)
MIN_CONVERGING_SIGNALS = 3        # Min dimensions deviating to trigger gate
CORRELATION_WINDOW_DAYS = 7       # Window to check cross-politician correlation
MIN_CORRELATED_POLITICIANS = 3    # Min politicians showing same shift

# --- Analysis Settings ---
SPACY_MODEL = "en_core_web_sm"

# --- Data Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASELINES_DIR = os.path.join(DATA_DIR, "baselines")
SPEECHES_DIR = os.path.join(DATA_DIR, "speeches")
SIGNALS_DIR = os.path.join(DATA_DIR, "signals")

for d in [DATA_DIR, BASELINES_DIR, SPEECHES_DIR, SIGNALS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Congress Settings ---
CURRENT_CONGRESS = 118  # 2023-2025
CHAMBERS = ["house", "senate"]
