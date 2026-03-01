"""
Layer 2: Baseline Profile Builder.

Constructs a psycholinguistic "fingerprint" for each politician based on 
their historical speech data. This baseline represents their "normal" 
linguistic behavior across all tracked dimensions.

The baseline includes:
- Mean and standard deviation for each dimension
- Per-topic baselines (how they talk about specific issues)
- Temporal trends (is their language shifting over time?)

Analogous to computing a price baseline + ATR in trading systems.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from analysis.psycholinguistic import PsycholinguisticAnalyzer


class PoliticianProfile:
    """
    Linguistic baseline profile for a single politician.
    
    Stores the statistical distribution of their psycholinguistic features
    across all analyzed speeches. Used as the comparison point for 
    anomaly detection.
    """
    
    def __init__(self, politician_id: str, name: str, party: str = "",
                 chamber: str = "", state: str = ""):
        self.politician_id = politician_id
        self.name = name
        self.party = party
        self.chamber = chamber
        self.state = state
        
        # Raw feature history: list of (date, features_dict) tuples
        self.speech_history: List[Tuple[str, Dict[str, float]]] = []
        
        # Computed baseline statistics
        self.baseline_mean: Dict[str, float] = {}
        self.baseline_std: Dict[str, float] = {}
        self.baseline_median: Dict[str, float] = {}
        self.baseline_q25: Dict[str, float] = {}
        self.baseline_q75: Dict[str, float] = {}
        
        # Per-topic baselines
        self.topic_baselines: Dict[str, Dict[str, float]] = {}
        
        # Metadata
        self.sample_count: int = 0
        self.last_updated: str = ""
        self.is_valid: bool = False
    
    def add_speech(self, date: str, features: Dict[str, float]):
        """Add a speech's feature vector to this politician's history."""
        self.speech_history.append((date, features))
        self.sample_count = len(self.speech_history)
    
    def compute_baseline(self, window_days: int = None):
        """
        Compute baseline statistics from speech history.
        
        Args:
            window_days: If set, only use speeches from the last N days.
                        If None, use all available speeches.
        """
        if len(self.speech_history) < config.MIN_SAMPLES_FOR_BASELINE:
            self.is_valid = False
            print(f"  [WARN] {self.name}: Only {len(self.speech_history)} samples "
                  f"(need {config.MIN_SAMPLES_FOR_BASELINE}). Baseline not valid.")
            # Still compute what we can for partial baselines
        
        # Filter by window if specified
        speeches = self.speech_history
        if window_days:
            cutoff = datetime.now() - pd.Timedelta(days=window_days)
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            speeches = [(d, f) for d, f in speeches if d >= cutoff_str]
        
        if not speeches:
            return
        
        # Build DataFrame from feature vectors
        feature_dicts = [f for _, f in speeches]
        df = pd.DataFrame(feature_dicts)
        
        # Compute statistics for each dimension
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64, float, int]:
                values = df[col].dropna()
                if len(values) > 0:
                    self.baseline_mean[col] = float(values.mean())
                    self.baseline_std[col] = float(values.std()) if len(values) > 1 else 0.0
                    self.baseline_median[col] = float(values.median())
                    self.baseline_q25[col] = float(values.quantile(0.25))
                    self.baseline_q75[col] = float(values.quantile(0.75))
        
        self.last_updated = datetime.now().isoformat()
        self.is_valid = len(speeches) >= config.MIN_SAMPLES_FOR_BASELINE
        
        if self.is_valid:
            print(f"  ✅ {self.name}: Baseline computed from {len(speeches)} speeches")
        else:
            print(f"  ⚠️  {self.name}: Partial baseline from {len(speeches)} speeches")
    
    def get_zscore(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute z-score deviation of a new speech from this politician's baseline.
        
        This is the core of the anomaly detection — how far is this speech
        from their "normal" on each dimension?
        
        Returns:
            Dict of dimension -> z-score. |z| > 2 is a strong signal.
        """
        zscores = {}
        for dim, value in features.items():
            if dim in self.baseline_mean and dim in self.baseline_std:
                mean = self.baseline_mean[dim]
                std = self.baseline_std[dim]
                if std > 0:
                    zscores[dim] = (value - mean) / std
                else:
                    # No variance — any deviation is notable
                    zscores[dim] = 0.0 if value == mean else (
                        2.0 if value > mean else -2.0
                    )
        return zscores
    
    def get_strongest_deviations(self, features: Dict[str, float], 
                                  top_n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Get the dimensions with strongest deviation from baseline.
        
        Returns:
            List of (dimension, z_score, direction) tuples, sorted by |z|.
        """
        zscores = self.get_zscore(features)
        
        deviations = []
        for dim, z in zscores.items():
            direction = "↑ ABOVE" if z > 0 else "↓ BELOW"
            deviations.append((dim, z, direction))
        
        deviations.sort(key=lambda x: abs(x[1]), reverse=True)
        return deviations[:top_n]
    
    def to_dict(self) -> dict:
        """Serialize profile for storage."""
        return {
            "politician_id": self.politician_id,
            "name": self.name,
            "party": self.party,
            "chamber": self.chamber,
            "state": self.state,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "is_valid": self.is_valid,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "baseline_median": self.baseline_median,
            "baseline_q25": self.baseline_q25,
            "baseline_q75": self.baseline_q75,
            "speech_history": [
                {"date": d, "features": f} for d, f in self.speech_history
            ],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PoliticianProfile":
        """Deserialize profile from storage."""
        profile = cls(
            politician_id=data["politician_id"],
            name=data["name"],
            party=data.get("party", ""),
            chamber=data.get("chamber", ""),
            state=data.get("state", ""),
        )
        profile.sample_count = data.get("sample_count", 0)
        profile.last_updated = data.get("last_updated", "")
        profile.is_valid = data.get("is_valid", False)
        profile.baseline_mean = data.get("baseline_mean", {})
        profile.baseline_std = data.get("baseline_std", {})
        profile.baseline_median = data.get("baseline_median", {})
        profile.baseline_q25 = data.get("baseline_q25", {})
        profile.baseline_q75 = data.get("baseline_q75", {})
        profile.speech_history = [
            (entry["date"], entry["features"]) 
            for entry in data.get("speech_history", [])
        ]
        return profile
    
    def save(self, directory: str = None):
        """Save profile to JSON file."""
        directory = directory or config.BASELINES_DIR
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.politician_id}.json")
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  💾 Saved profile: {filepath}")
    
    @classmethod
    def load(cls, politician_id: str, directory: str = None) -> Optional["PoliticianProfile"]:
        """Load profile from JSON file."""
        directory = directory or config.BASELINES_DIR
        filepath = os.path.join(directory, f"{politician_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class ProfileBuilder:
    """
    Orchestrates baseline construction for multiple politicians.
    
    Takes a collection of SpeechRecords, groups them by speaker,
    analyzes each speech, and builds PoliticianProfile objects.
    """
    
    def __init__(self):
        self.analyzer = PsycholinguisticAnalyzer()
        self.profiles: Dict[str, PoliticianProfile] = {}
    
    def build_profiles(self, speeches: list, 
                       window_days: int = None) -> Dict[str, PoliticianProfile]:
        """
        Build baseline profiles from a list of SpeechRecords.
        
        Args:
            speeches: List of SpeechRecord objects
            window_days: Optional rolling window for baseline calculation
        
        Returns:
            Dict of politician_id -> PoliticianProfile
        """
        print(f"\n{'=' * 60}")
        print(f"  BUILDING BASELINE PROFILES")
        print(f"  {len(speeches)} speeches to process")
        print(f"{'=' * 60}\n")
        
        # Group speeches by speaker
        by_speaker = {}
        for speech in speeches:
            sid = speech.speaker_id
            if sid not in by_speaker:
                by_speaker[sid] = {
                    "name": speech.speaker_name,
                    "party": speech.party,
                    "chamber": speech.chamber,
                    "state": speech.state,
                    "speeches": [],
                }
            by_speaker[sid]["speeches"].append(speech)
        
        print(f"  Found {len(by_speaker)} unique speakers\n")
        
        # Analyze each speaker's speeches
        for speaker_id, info in by_speaker.items():
            print(f"  📝 Processing: {info['name']} ({len(info['speeches'])} speeches)")
            
            profile = PoliticianProfile(
                politician_id=speaker_id,
                name=info["name"],
                party=info["party"],
                chamber=info["chamber"],
                state=info["state"],
            )
            
            for speech in info["speeches"]:
                # Skip very short speeches
                if speech.word_count < config.MIN_WORD_COUNT:
                    continue
                
                features = self.analyzer.analyze(speech.text)
                if features:
                    profile.add_speech(speech.date, features)
            
            # Compute baseline statistics
            profile.compute_baseline(window_days)
            self.profiles[speaker_id] = profile
        
        print(f"\n  {'=' * 60}")
        valid = sum(1 for p in self.profiles.values() if p.is_valid)
        print(f"  ✅ {valid} valid baselines out of {len(self.profiles)} politicians")
        print(f"  {'=' * 60}\n")
        
        return self.profiles
    
    def save_all_profiles(self, directory: str = None):
        """Save all profiles to disk."""
        for profile in self.profiles.values():
            profile.save(directory)
    
    def load_all_profiles(self, directory: str = None) -> Dict[str, PoliticianProfile]:
        """Load all profiles from disk."""
        directory = directory or config.BASELINES_DIR
        if not os.path.exists(directory):
            return {}
        
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                politician_id = filename[:-5]
                profile = PoliticianProfile.load(politician_id, directory)
                if profile:
                    self.profiles[politician_id] = profile
        
        return self.profiles
    
    def print_profile_summary(self, politician_id: str):
        """Print a human-readable summary of a politician's baseline."""
        profile = self.profiles.get(politician_id)
        if not profile:
            print(f"No profile found for {politician_id}")
            return
        
        print(f"\n{'=' * 60}")
        print(f"  BASELINE PROFILE: {profile.name}")
        print(f"  {profile.party} | {profile.chamber} | {profile.state}")
        print(f"  Speeches analyzed: {profile.sample_count}")
        print(f"  Valid: {'✅ Yes' if profile.is_valid else '⚠️  No (insufficient data)'}")
        print(f"{'=' * 60}")
        
        if not profile.baseline_mean:
            print("  No baseline data computed yet.")
            return
        
        # Key dimensions
        key_dims = [
            ("authenticity_score", "Authenticity"),
            ("clout_score", "Clout"),
            ("cognitive_complexity", "Cognitive Complexity"),
            ("emotional_tone", "Emotional Tone"),
            ("certainty_hedging_ratio", "Certainty/Hedging Ratio"),
            ("temporal_orientation", "Temporal Orientation"),
            ("i_words", "I-words %"),
            ("we_words", "We-words %"),
            ("certainty", "Certainty %"),
            ("hedging", "Hedging %"),
            ("anger", "Anger %"),
            ("positive_emotion", "Positive Emotion %"),
            ("power", "Power Words %"),
        ]
        
        print(f"\n  {'Dimension':<25} {'Mean':>8} {'Std Dev':>8} {'Range (IQR)':>15}")
        print(f"  {'—' * 58}")
        
        for dim, label in key_dims:
            mean = profile.baseline_mean.get(dim, 0)
            std = profile.baseline_std.get(dim, 0)
            q25 = profile.baseline_q25.get(dim, 0)
            q75 = profile.baseline_q75.get(dim, 0)
            print(f"  {label:<25} {mean:>8.2f} {std:>8.2f} {q25:>6.2f} – {q75:>6.2f}")
        
        print()
