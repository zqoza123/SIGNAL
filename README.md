# Political Conviction Signal Engine (PCSE)

## Overview
A psycholinguistic NLP pipeline that detects **conviction signals** in U.S. political speech 
by measuring deviations from each politician's established linguistic baseline. Designed to 
generate tradeable signals for prediction markets (Kalshi, Polymarket).

## Research Foundation
Built on James Pennebaker's LIWC framework and decades of psycholinguistic research:
- **Newman et al. (2003)** — Deception detection via linguistic markers
- **Pennebaker & Lay (2002)** — Language use and personality during crises (Giuliani study)
- **Kacewicz et al. (2014)** — Clout and social dominance in language
- **Tausczik & Pennebaker (2010)** — Psychological meaning of words
- **Gerrish & Blei (2011)** — Predicting legislative roll calls from text

## Architecture

### Layer 1: Data Ingestion (`ingestion/`)
- Congress.gov API → floor speeches, committee remarks
- Congressional Record daily digest
- Politician social media feeds (future)

### Layer 2: Baseline Construction (`baseline/`)
- Per-politician linguistic fingerprints across 25+ psycholinguistic dimensions
- Rolling baseline with configurable window (default: 90 days)
- Minimum sample threshold for statistical validity

### Layer 3: Anomaly Detection (`detection/`)
- Z-score deviation from personal baseline on each dimension
- Multi-signal convergence gating (inspired by RSI divergence detection)
- Cross-politician correlation detection (coordinated messaging)

### Layer 4: Topic-Signal Mapping (`mapping/`)
- BERTopic / LDA topic extraction from gated signals
- Map policy topics to prediction market contract taxonomy

### Layer 5: Market Integration (future)
- Kalshi/Polymarket API integration
- Signal strength vs. implied probability comparison

## Psycholinguistic Dimensions Tracked

### Function Word Analysis (via spaCy POS tagging)
| Dimension | What It Measures | Signal When Shifts |
|-----------|-----------------|-------------------|
| I-words (I, me, my) | Self-focus, personal investment | ↑ = personal conviction, ↓ = distancing |
| We-words (we, us, our) | Coalition/group identity | ↑ = coalition forming |
| They-words (they, them) | Out-group framing | ↑ = adversarial positioning |
| Articles (a, an, the) | Concrete/categorical thinking | ↑ = analytical mode |
| Prepositions | Cognitive complexity | ↑ = nuanced thought |
| Auxiliary verbs | Hedging/certainty | Patterns indicate commitment level |

### Custom Lexicon Analysis
| Dimension | Examples | Research Basis |
|-----------|---------|---------------|
| Certainty | "absolutely", "definitely", "without question" | Newman et al. 2003 |
| Hedging | "perhaps", "might", "it seems" | Pennebaker 2011 |
| Negation | "not", "never", "no", "cannot" | Newman et al. 2003 |
| Causation | "because", "therefore", "consequently" | Tausczik & Pennebaker 2010 |
| Future-focus | "will", "shall", "going to" | Temporal orientation research |
| Past-focus | "was", "had", "were" | Temporal orientation research |
| Achievement | "win", "succeed", "accomplish" | LIWC category |
| Power/dominance | "control", "force", "demand" | Kacewicz et al. 2014 |
| Anger | "hate", "destroy", "attack" | LIWC emotion categories |
| Anxiety | "worry", "fear", "nervous" | LIWC emotion categories |

### Composite Scores (Derived)
- **Authenticity Score** — Based on pronoun patterns + spontaneity markers
- **Clout Score** — Based on we-words, power language, certainty
- **Cognitive Complexity** — Based on conjunctions, prepositions, exclusive words
- **Emotional Volatility** — Rolling std dev of emotional tone

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Set environment variables
export CONGRESS_API_KEY="your_key_here"

# Run baseline construction
python main.py --mode baseline --congress 118

# Run anomaly detection on new data
python main.py --mode detect --days 7
```

## Project Structure
```
political-signal-engine/
├── main.py                     # CLI orchestrator
├── config.py                   # Configuration and API keys
├── requirements.txt
├── ingestion/
│   ├── congress_api.py         # Congress.gov API client
│   └── text_preprocessor.py   # Clean and normalize text
├── analysis/
│   ├── psycholinguistic.py    # Core LIWC-style analyzer
│   └── lexicons.py            # Custom word dictionaries
├── baseline/
│   └── profile_builder.py     # Per-politician baseline construction
├── detection/
│   └── anomaly_detector.py    # Gating model for signal detection
├── mapping/
│   └── topic_mapper.py        # Topic extraction and contract mapping
├── utils/
│   └── helpers.py             # Shared utilities
├── data/                       # Local data storage
└── tests/
    └── test_analyzer.py       # Unit tests
```

## Author
David — Ohio State University CS '26
Built as part of quantitative finance research combining NLP, 
psycholinguistics, and prediction market signal generation.
