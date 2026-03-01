"""
Core Psycholinguistic Analyzer.

Replicates key LIWC dimensions using spaCy POS tagging + custom lexicons.
Produces a feature vector for any input text that captures:
- Function word patterns (pronouns, articles, prepositions)
- Lexicon-based scores (certainty, hedging, emotion, etc.)
- Composite scores (authenticity, clout, cognitive complexity)
- Structural features (sentence length, word count, vocabulary richness)

Based on: Pennebaker (2011), Newman et al. (2003), Tausczik & Pennebaker (2010)
"""

import re
from collections import Counter
from typing import Dict, Optional

import numpy as np

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from analysis.lexicons import ALL_LEXICONS, PROFILE_DIMENSIONS


class PsycholinguisticAnalyzer:
    """
    Analyzes text across psycholinguistic dimensions.
    
    Produces a normalized feature vector for each text sample,
    enabling comparison across speeches of different lengths.
    """
    
    # --- Pronoun Categories (from Pennebaker's function word research) ---
    I_WORDS = {"i", "me", "my", "mine", "myself"}
    WE_WORDS = {"we", "us", "our", "ours", "ourselves"}
    YOU_WORDS = {"you", "your", "yours", "yourself", "yourselves"}
    THEY_WORDS = {"they", "them", "their", "theirs", "themselves"}
    HE_SHE_WORDS = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
    
    # --- Articles ---
    ARTICLES = {"a", "an", "the"}
    
    # --- Common Prepositions ---
    PREPOSITIONS = {
        "about", "above", "across", "after", "against", "along", "among",
        "around", "at", "before", "behind", "below", "beneath", "beside",
        "between", "beyond", "by", "down", "during", "except", "for",
        "from", "in", "inside", "into", "near", "of", "off", "on",
        "onto", "out", "outside", "over", "past", "through", "to",
        "toward", "towards", "under", "underneath", "until", "up",
        "upon", "with", "within", "without",
    }
    
    # --- Conjunctions ---
    CONJUNCTIONS = {
        "and", "but", "or", "nor", "for", "yet", "so",
        "although", "because", "since", "unless", "while",
        "whereas", "however", "therefore", "moreover", "furthermore",
        "nevertheless", "nonetheless", "meanwhile",
    }
    
    # --- Auxiliary/Modal Verbs ---
    AUXILIARY_VERBS = {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might",
        "can", "could", "must", "need", "dare", "ought",
    }
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize analyzer with spaCy model for POS tagging."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                # Disable unnecessary pipeline components for speed
                self.nlp.select_pipes(enable=["tagger", "attribute_ruler", "lemmatizer"])
            except OSError:
                print(f"[WARN] spaCy model '{spacy_model}' not found. "
                      "Run: python -m spacy download en_core_web_sm")
                print("[WARN] Falling back to regex-based tokenization.")
    
    def tokenize(self, text: str) -> list:
        """Simple whitespace + punctuation tokenizer."""
        # Lowercase and split on non-alphanumeric (keeping apostrophes for contractions)
        tokens = re.findall(r"[a-z']+", text.lower())
        return [t for t in tokens if len(t) > 0]
    
    def analyze(self, text: str) -> Optional[Dict[str, float]]:
        """
        Produce a full psycholinguistic feature vector for a text.
        
        Returns dict of dimension_name -> normalized_score (% of total words),
        plus composite scores and structural features.
        Returns None if text is too short for reliable analysis.
        """
        tokens = self.tokenize(text)
        word_count = len(tokens)
        
        if word_count < 50:
            return None  # Too short for reliable psycholinguistic analysis
        
        token_set = set(tokens)
        token_counter = Counter(tokens)
        
        features = {}
        
        # =====================================================================
        # 1. FUNCTION WORD ANALYSIS (% of total words)
        # =====================================================================
        features["i_words"] = self._count_category(tokens, self.I_WORDS) / word_count * 100
        features["we_words"] = self._count_category(tokens, self.WE_WORDS) / word_count * 100
        features["you_words"] = self._count_category(tokens, self.YOU_WORDS) / word_count * 100
        features["they_words"] = self._count_category(tokens, self.THEY_WORDS) / word_count * 100
        features["shehe_words"] = self._count_category(tokens, self.HE_SHE_WORDS) / word_count * 100
        features["articles"] = self._count_category(tokens, self.ARTICLES) / word_count * 100
        features["prepositions"] = self._count_category(tokens, self.PREPOSITIONS) / word_count * 100
        features["conjunctions"] = self._count_category(tokens, self.CONJUNCTIONS) / word_count * 100
        features["auxiliary_verbs"] = self._count_category(tokens, self.AUXILIARY_VERBS) / word_count * 100
        
        # Total pronouns
        all_pronouns = (self.I_WORDS | self.WE_WORDS | self.YOU_WORDS | 
                       self.THEY_WORDS | self.HE_SHE_WORDS)
        features["total_pronouns"] = self._count_category(tokens, all_pronouns) / word_count * 100
        
        # =====================================================================
        # 2. LEXICON-BASED SCORES (% of total words)
        # =====================================================================
        for dimension in PROFILE_DIMENSIONS:
            lexicon = ALL_LEXICONS[dimension]
            features[dimension] = self._count_lexicon(tokens, lexicon) / word_count * 100
        
        # =====================================================================
        # 3. STRUCTURAL FEATURES
        # =====================================================================
        features["word_count"] = word_count
        
        # Sentence count (approximate)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        features["sentence_count"] = len(sentences)
        features["words_per_sentence"] = word_count / max(len(sentences), 1)
        
        # Vocabulary richness (type-token ratio)
        unique_words = len(set(tokens))
        features["type_token_ratio"] = unique_words / word_count
        
        # Average word length
        features["avg_word_length"] = np.mean([len(t) for t in tokens])
        
        # Long words (>6 chars) — indicates formal/complex language
        long_words = sum(1 for t in tokens if len(t) > 6)
        features["long_words_pct"] = long_words / word_count * 100
        
        # =====================================================================
        # 4. COMPOSITE SCORES (derived from Pennebaker's summary variables)
        # =====================================================================
        features["authenticity_score"] = self._compute_authenticity(features)
        features["clout_score"] = self._compute_clout(features)
        features["cognitive_complexity"] = self._compute_cognitive_complexity(features)
        features["emotional_tone"] = self._compute_emotional_tone(features)
        features["certainty_hedging_ratio"] = self._compute_certainty_ratio(features)
        features["temporal_orientation"] = self._compute_temporal_orientation(features)
        
        return features
    
    def _count_category(self, tokens: list, category: set) -> int:
        """Count tokens that match a category set."""
        return sum(1 for t in tokens if t in category)
    
    def _count_lexicon(self, tokens: list, lexicon: set) -> int:
        """
        Count tokens matching a lexicon, including multi-word expressions.
        For single words, does direct lookup.
        For multi-word expressions, joins adjacent tokens and checks.
        """
        count = 0
        single_words = {w for w in lexicon if " " not in w}
        multi_words = {w for w in lexicon if " " in w}
        
        # Single word matches
        for token in tokens:
            if token in single_words:
                count += 1
        
        # Multi-word matches (bigrams/trigrams)
        if multi_words:
            text_lower = " ".join(tokens)
            for phrase in multi_words:
                count += text_lower.count(phrase)
        
        return count
    
    def _compute_authenticity(self, features: dict) -> float:
        """
        Approximate LIWC Authenticity score.
        
        Based on Newman et al. (2003): Authentic/honest speech has:
        - MORE first-person singular pronouns (I, me, my)
        - MORE exclusive words (but, except, without)
        - MORE negative emotion words
        - FEWER motion words and fewer negations
        
        High authenticity = spontaneous, unguarded
        Low authenticity = prepared, self-monitoring, potentially deceptive
        
        Returns score 0-100 (higher = more authentic).
        """
        # Positive contributors to authenticity
        auth = 0.0
        auth += features.get("i_words", 0) * 8.0       # I-words strongly indicate authenticity
        auth += features.get("exclusive", 0) * 5.0      # Exclusive words = differentiated thinking
        auth += features.get("negation", 0) * (-3.0)    # High negation = less authentic
        auth -= features.get("we_words", 0) * 2.0       # We-words can indicate performative speech
        auth -= features.get("long_words_pct", 0) * 0.3 # Complex words = more prepared
        
        # Normalize to 0-100 range (approximate)
        auth = max(0, min(100, 50 + auth * 3))
        return round(auth, 2)
    
    def _compute_clout(self, features: dict) -> float:
        """
        Approximate LIWC Clout score.
        
        Based on Kacewicz et al. (2014): High-status speakers use:
        - MORE we-words and you-words
        - FEWER I-words
        - MORE certainty language
        - MORE power/authority words
        
        Returns score 0-100 (higher = more authoritative).
        """
        clout = 0.0
        clout += features.get("we_words", 0) * 6.0
        clout += features.get("you_words", 0) * 3.0
        clout -= features.get("i_words", 0) * 5.0
        clout += features.get("certainty", 0) * 4.0
        clout += features.get("power", 0) * 5.0
        clout -= features.get("hedging", 0) * 3.0
        
        clout = max(0, min(100, 50 + clout * 2.5))
        return round(clout, 2)
    
    def _compute_cognitive_complexity(self, features: dict) -> float:
        """
        Measure of how complex/nuanced the thinking is.
        
        High complexity: lots of conjunctions, exclusive words, insight words,
        causation words, and higher type-token ratio.
        Low complexity: simple, declarative, absolutist.
        
        Returns score 0-100.
        """
        complexity = 0.0
        complexity += features.get("conjunctions", 0) * 4.0
        complexity += features.get("exclusive", 0) * 5.0
        complexity += features.get("insight", 0) * 4.0
        complexity += features.get("causation", 0) * 5.0
        complexity += features.get("type_token_ratio", 0) * 20.0
        complexity += features.get("prepositions", 0) * 1.5
        complexity -= features.get("certainty", 0) * 2.0  # Certainty reduces nuance
        
        complexity = max(0, min(100, complexity * 2))
        return round(complexity, 2)
    
    def _compute_emotional_tone(self, features: dict) -> float:
        """
        Net emotional valence.
        
        Positive = more positive emotion words, fewer anger/anxiety.
        Negative = more anger, anxiety, fewer positive words.
        
        Returns score 0-100 (50 = neutral, >50 = positive, <50 = negative).
        """
        pos = features.get("positive_emotion", 0)
        neg = features.get("anger", 0) + features.get("anxiety", 0)
        
        tone = 50 + (pos - neg) * 10
        tone = max(0, min(100, tone))
        return round(tone, 2)
    
    def _compute_certainty_ratio(self, features: dict) -> float:
        """
        Ratio of certainty to hedging language.
        
        > 1.0 = more certain than hedging (conviction)
        < 1.0 = more hedging than certainty (deliberation/deception)
        = 1.0 = balanced
        
        Returns raw ratio (not bounded 0-100).
        """
        cert = features.get("certainty", 0)
        hedge = features.get("hedging", 0)
        
        if hedge == 0:
            return cert * 10 if cert > 0 else 1.0
        return round(cert / hedge, 4)
    
    def _compute_temporal_orientation(self, features: dict) -> float:
        """
        Where is the speaker oriented in time?
        
        Positive = future-oriented (commitment, action)
        Negative = past-oriented (reflection, defense)
        Zero = present-focused
        
        Returns score from -100 to 100.
        """
        future = features.get("future", 0)
        past = features.get("past", 0)
        present = features.get("present", 0)
        
        total = future + past + present
        if total == 0:
            return 0.0
        
        # Future - Past, normalized
        orientation = ((future - past) / total) * 100
        return round(orientation, 2)
    
    def get_feature_names(self) -> list:
        """Return ordered list of all feature names this analyzer produces."""
        # Run on dummy text to get all keys
        dummy_features = {
            # Function words
            "i_words": 0, "we_words": 0, "you_words": 0, "they_words": 0,
            "shehe_words": 0, "articles": 0, "prepositions": 0,
            "conjunctions": 0, "auxiliary_verbs": 0, "total_pronouns": 0,
            # Lexicon dimensions
            "certainty": 0, "hedging": 0, "negation": 0, "exclusive": 0,
            "causation": 0, "insight": 0, "future": 0, "past": 0,
            "present": 0, "anger": 0, "anxiety": 0, "positive_emotion": 0,
            "power": 0, "achievement": 0, "economic": 0,
            # Structural
            "word_count": 0, "sentence_count": 0, "words_per_sentence": 0,
            "type_token_ratio": 0, "avg_word_length": 0, "long_words_pct": 0,
            # Composites
            "authenticity_score": 0, "clout_score": 0,
            "cognitive_complexity": 0, "emotional_tone": 0,
            "certainty_hedging_ratio": 0, "temporal_orientation": 0,
        }
        return list(dummy_features.keys())


def demo():
    """Run a quick demo analysis on sample political texts."""
    analyzer = PsycholinguisticAnalyzer()
    
    # Sample: High certainty, high clout speech
    speech_a = """
    We will absolutely deliver on our promise to the American people. 
    This is not a question of if, but when. Our administration has 
    committed to ensuring every family in this nation has access to 
    affordable healthcare. We must act now, and we will act decisively. 
    The time for deliberation is over. We have the votes, we have the 
    plan, and we will execute it without hesitation. The American people 
    demand action, and we will not let them down. Together, we will 
    build a stronger, more prosperous nation for our children and 
    grandchildren. This is our moment, and we will seize it.
    """
    
    # Sample: High hedging, low certainty speech
    speech_b = """
    I think we might need to reconsider some aspects of this proposal. 
    It seems like there could be unintended consequences that perhaps 
    we haven't fully explored. I believe we should probably take more 
    time to understand the implications. Some people have suggested 
    that this approach might not work as well as we hope. I'm not 
    entirely sure where I stand on this yet, but I feel we should 
    maybe look at alternative approaches. It appears that the situation 
    is more complex than we initially thought. I would suggest we 
    consider forming a committee to study this further before making 
    any final decisions on the matter.
    """
    
    print("=" * 70)
    print("POLITICAL CONVICTION SIGNAL ENGINE — Demo Analysis")
    print("=" * 70)
    
    for label, speech in [("HIGH CONVICTION", speech_a), ("HEDGING/UNCERTAIN", speech_b)]:
        print(f"\n{'—' * 70}")
        print(f"  {label}")
        print(f"{'—' * 70}")
        features = analyzer.analyze(speech)
        
        if features:
            # Key indicators
            print(f"\n  📊 COMPOSITE SCORES:")
            print(f"     Authenticity:       {features['authenticity_score']:.1f} / 100")
            print(f"     Clout:              {features['clout_score']:.1f} / 100")
            print(f"     Cognitive Complex:  {features['cognitive_complexity']:.1f} / 100")
            print(f"     Emotional Tone:     {features['emotional_tone']:.1f} / 100")
            print(f"     Certainty/Hedging:  {features['certainty_hedging_ratio']:.2f}")
            print(f"     Temporal Orient:    {features['temporal_orientation']:.1f}")
            
            print(f"\n  🔤 PRONOUN PROFILE:")
            print(f"     I-words:   {features['i_words']:.2f}%")
            print(f"     We-words:  {features['we_words']:.2f}%")
            print(f"     You-words: {features['you_words']:.2f}%")
            print(f"     They:      {features['they_words']:.2f}%")
            
            print(f"\n  ⚡ CONVICTION SIGNALS:")
            print(f"     Certainty:      {features['certainty']:.2f}%")
            print(f"     Hedging:        {features['hedging']:.2f}%")
            print(f"     Negation:       {features['negation']:.2f}%")
            print(f"     Power words:    {features['power']:.2f}%")
            print(f"     Future-focus:   {features['future']:.2f}%")
            
            print(f"\n  📏 STRUCTURAL:")
            print(f"     Word count:     {features['word_count']}")
            print(f"     Words/sentence: {features['words_per_sentence']:.1f}")
            print(f"     Vocab richness: {features['type_token_ratio']:.3f}")
    
    print(f"\n{'=' * 70}")
    print("Demo complete. These two speeches demonstrate how the analyzer")
    print("differentiates conviction from deliberation/uncertainty.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo()
