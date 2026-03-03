"""
Layer 3: Anomaly Detection — The Gating Model.

Detects when a politician's speech significantly deviates from their 
established linguistic baseline. Implements a multi-signal convergence 
gate inspired by algorithmic trading signal detection.

The gate opens (fires a signal) when:
1. Multiple psycholinguistic dimensions deviate simultaneously
2. The deviation is statistically significant (|z| > threshold)
3. Cross-politician correlation is detected (coordinated shifts)

Architecture parallels:
- Z-score deviation → ATR-based volatility in price
- Multi-signal convergence → RSI divergence confirmation
- Cross-politician correlation → correlated volume across instruments
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

import config
from analysis.psycholinguistic import PsycholinguisticAnalyzer
from analysis.lexicons import POLICY_DOMAINS, ALL_LEXICONS
from baseline.profile_builder import PoliticianProfile


@dataclass
class Signal:
    """
    A detected conviction signal.
    
    Represents a statistically significant deviation in a politician's
    language that may indicate a shift in conviction, stance, or
    coordinated party strategy.
    """
    politician_id: str
    politician_name: str
    party: str
    date: str
    signal_type: str          # 'conviction', 'hedging', 'coalition', 'off_script', 'coordinated'
    strength: float           # 0-1 normalized signal strength
    
    # Deviation details
    top_deviations: List[Tuple[str, float, str]] = field(default_factory=list)
    converging_count: int = 0   # How many dimensions are deviating
    
    # Context
    policy_domains: List[str] = field(default_factory=list)
    speech_snippet: str = ""
    
    # For coordinated signals
    correlated_politicians: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return (f"Signal({self.signal_type.upper()}, strength={self.strength:.2f}, "
                f"{self.politician_name}, {self.date}, "
                f"domains={self.policy_domains})")
    
    def summary(self) -> str:
        """Human-readable signal summary."""
        lines = [
            f"{'🔴' if self.strength > 0.7 else '🟡' if self.strength > 0.4 else '🟢'} "
            f"{self.signal_type.upper()} SIGNAL — {self.politician_name} ({self.party})",
            f"   Date: {self.date}",
            f"   Strength: {self.strength:.2f} / 1.00",
            f"   Converging dimensions: {self.converging_count}",
        ]
        
        if self.policy_domains:
            lines.append(f"   Policy domains: {', '.join(self.policy_domains)}")
        
        if self.top_deviations:
            lines.append(f"   Top deviations:")
            for dim, z, direction in self.top_deviations[:5]:
                lines.append(f"     {direction} {dim}: z={z:+.2f}")
        
        if self.correlated_politicians:
            lines.append(f"   Correlated with: {', '.join(self.correlated_politicians)}")
        
        if self.speech_snippet:
            snippet = self.speech_snippet[:150] + "..." if len(self.speech_snippet) > 150 else self.speech_snippet
            lines.append(f"   Snippet: \"{snippet}\"")
        
        return "\n".join(lines)


class AnomalyDetector:
    """
    The gating model for political conviction signals.
    
    Compares new speeches against politician baselines and fires
    signals when statistically significant deviations are detected.
    """
    
    def __init__(self, profiles: Dict[str, PoliticianProfile]):
        self.profiles = profiles
        self.analyzer = PsycholinguisticAnalyzer()
        self.recent_signals: List[Signal] = []
        
        # Dimensions to monitor for each signal type
        self.conviction_dims = [
            "certainty", "hedging", "certainty_hedging_ratio",
            "clout_score", "power", "future",
        ]
        self.authenticity_dims = [
            "authenticity_score", "i_words", "we_words",
            "words_per_sentence", "type_token_ratio",
        ]
        self.emotion_dims = [
            "anger", "anxiety", "positive_emotion",
            "emotional_tone", "negation",
        ]
        self.complexity_dims = [
            "cognitive_complexity", "exclusive", "causation",
            "insight", "conjunctions",
        ]
    
    def analyze_speech(self, speech) -> Optional[Signal]:
        """
        Analyze a single speech against the speaker's baseline.
        
        Args:
            speech: SpeechRecord object
        
        Returns:
            Signal if gate fires, None otherwise
        """
        profile = self.profiles.get(speech.speaker_id)
        if not profile or not profile.baseline_mean:
            return None
        
        # Analyze the speech
        features = self.analyzer.analyze(speech.text)
        if not features:
            return None
        
        # Compute z-scores against baseline
        zscores = profile.get_zscore(features)
        
        if not zscores:
            return None
        
        # Check for multi-signal convergence
        strong_deviations = {
            dim: z for dim, z in zscores.items() 
            if abs(z) >= config.ZSCORE_THRESHOLD_MODERATE
        }
        
        very_strong = {
            dim: z for dim, z in zscores.items()
            if abs(z) >= config.ZSCORE_THRESHOLD_STRONG
        }
        
        converging_count = len(strong_deviations)
        
        # Gate check: do we have enough converging signals?
        if converging_count < config.MIN_CONVERGING_SIGNALS and len(very_strong) < 2:
            return None  # Gate stays closed
        
        # === GATE IS OPEN — classify the signal ===
        
        # Determine signal type
        signal_type = self._classify_signal(zscores, features)
        
        # Compute signal strength (0-1)
        strength = self._compute_strength(zscores, converging_count)
        
        # Get top deviations
        top_devs = profile.get_strongest_deviations(features, top_n=7)
        
        # Detect policy domains
        domains = self._detect_policy_domains(features)
        
        signal = Signal(
            politician_id=speech.speaker_id,
            politician_name=speech.speaker_name,
            party=speech.party,
            date=speech.date,
            signal_type=signal_type,
            strength=strength,
            top_deviations=top_devs,
            converging_count=converging_count,
            policy_domains=domains,
            speech_snippet=speech.text[:300],
        )
        
        self.recent_signals.append(signal)
        return signal
    
    def analyze_batch(self, speeches: list) -> List[Signal]:
        """
        Analyze a batch of speeches and detect signals.
        Also checks for cross-politician correlation.
        """
        signals = []
        
        for speech in speeches:
            signal = self.analyze_speech(speech)
            if signal:
                signals.append(signal)
        
        # Check for coordinated signals
        coordinated = self._detect_coordination(signals)
        signals.extend(coordinated)
        
        return signals
    
    def _classify_signal(self, zscores: dict, features: dict) -> str:
        """
        Classify what type of signal the deviations represent.
        
        Signal types:
        - conviction: Speaker is showing unusual certainty/commitment
        - hedging: Speaker is unusually uncertain (may signal retreat)
        - off_script: Authenticity spike (going off prepared remarks)
        - emotional: Unusual emotional intensity
        - complexity_shift: Cognitive complexity changed significantly
        """
        # Check conviction dimensions
        conviction_zs = [zscores.get(d, 0) for d in self.conviction_dims if d in zscores]
        avg_conviction = np.mean(conviction_zs) if conviction_zs else 0
        
        # Check authenticity dimensions
        auth_zs = [zscores.get(d, 0) for d in self.authenticity_dims if d in zscores]
        avg_auth = np.mean(auth_zs) if auth_zs else 0
        
        # Check emotion dimensions
        emotion_zs = [abs(zscores.get(d, 0)) for d in self.emotion_dims if d in zscores]
        avg_emotion = np.mean(emotion_zs) if emotion_zs else 0
        
        # Classify based on strongest signal cluster
        if avg_conviction > 1.5:
            return "conviction"
        elif avg_conviction < -1.5:
            return "hedging"
        elif abs(avg_auth) > 1.5:
            return "off_script"
        elif avg_emotion > 1.5:
            return "emotional"
        else:
            return "complexity_shift"
    
    def _compute_strength(self, zscores: dict, converging_count: int) -> float:
        """
        Compute overall signal strength from 0 to 1.
        
        Combines:
        - Maximum |z-score| across dimensions
        - Number of converging signals
        - Average deviation magnitude
        """
        if not zscores:
            return 0.0
        
        abs_zscores = [abs(z) for z in zscores.values()]
        max_z = max(abs_zscores)
        avg_z = np.mean(abs_zscores)
        
        # Normalize components
        z_component = min(max_z / 4.0, 1.0)        # Max z of 4 → strength 1.0
        converge_component = min(converging_count / 8.0, 1.0)  # 8 converging → 1.0
        avg_component = min(avg_z / 2.0, 1.0)      # Avg z of 2 → 1.0
        
        # Weighted combination
        strength = (z_component * 0.4 + 
                   converge_component * 0.35 + 
                   avg_component * 0.25)
        
        return round(min(strength, 1.0), 4)
    
    def _detect_policy_domains(self, features: dict) -> List[str]:
        """
        Detect which policy domains are relevant to this speech.
        
        Uses lexicon hit rates to identify primary topics.
        """
        domains = []
        for domain in POLICY_DOMAINS:
            score = features.get(domain, 0)
            if score > 0.3:  # At least 0.3% of words match domain lexicon
                domains.append(domain)
        
        # Sort by relevance
        domains.sort(key=lambda d: features.get(d, 0), reverse=True)
        return domains[:3]  # Top 3 domains
    
    def _detect_coordination(self, signals: List[Signal]) -> List[Signal]:
        """
        Detect coordinated language shifts across multiple politicians.
        
        When multiple politicians from the same party show similar 
        deviations on the same topic within a short time window,
        it suggests coordinated messaging (party decision made).
        """
        coordinated_signals = []
        
        if len(signals) < config.MIN_CORRELATED_POLITICIANS:
            return coordinated_signals
        
        # Group signals by party + policy domain
        from collections import defaultdict
        party_domain_groups = defaultdict(list)
        
        for signal in signals:
            for domain in signal.policy_domains:
                key = f"{signal.party}|{domain}"
                party_domain_groups[key].append(signal)
        
        # Check each group for coordination
        for key, group_signals in party_domain_groups.items():
            if len(group_signals) >= config.MIN_CORRELATED_POLITICIANS:
                party, domain = key.split("|")
                
                # Check if signals are in the same direction
                signal_types = [s.signal_type for s in group_signals]
                dominant_type = max(set(signal_types), key=signal_types.count)
                same_direction = signal_types.count(dominant_type) / len(signal_types)
                
                if same_direction >= 0.6:  # 60%+ agreement
                    avg_strength = np.mean([s.strength for s in group_signals])
                    politician_names = [s.politician_name for s in group_signals]
                    
                    coord_signal = Signal(
                        politician_id="COORDINATED",
                        politician_name=f"{party} Caucus",
                        party=party,
                        date=group_signals[0].date,
                        signal_type="coordinated",
                        strength=min(avg_strength * 1.3, 1.0),  # Boost for coordination
                        policy_domains=[domain],
                        converging_count=len(group_signals),
                        correlated_politicians=politician_names,
                    )
                    coordinated_signals.append(coord_signal)
        
        return coordinated_signals
    
    def get_signal_report(self, signals: List[Signal] = None) -> str:
        """Generate a formatted report of detected signals."""
        signals = signals or self.recent_signals
        
        if not signals:
            return "No signals detected."
        
        lines = [
            f"\n{'=' * 70}",
            f"  SIGNAL REPORT — {len(signals)} signals detected",
            f"{'=' * 70}",
        ]
        
        # Sort by strength
        signals_sorted = sorted(signals, key=lambda s: s.strength, reverse=True)
        
        for i, signal in enumerate(signals_sorted, 1):
            lines.append(f"\n  Signal #{i}")
            lines.append(f"  {'—' * 60}")
            lines.append(signal.summary())
        
        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)
