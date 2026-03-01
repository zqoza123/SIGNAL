"""
Custom Psycholinguistic Lexicons for Political Speech Analysis.

Built from published research by Pennebaker, Newman, Tausczik et al.
These replicate core LIWC categories using open-source word lists derived
from published category descriptions and validated against Empath correlations.

References:
- Newman et al. (2003) "Lying Words" - deception markers
- Pennebaker (2011) "The Secret Life of Pronouns"
- Tausczik & Pennebaker (2010) "Psychological Meaning of Words"
- Kacewicz et al. (2014) - Clout/social hierarchy markers
"""


# =============================================================================
# CERTAINTY vs HEDGING (Newman et al., 2003; Pennebaker, 2011)
# =============================================================================
# High certainty = speaker has conviction, has made a decision
# High hedging = speaker is deliberating, uncertain, or being deceptive

CERTAINTY_WORDS = {
    # Absolute certainty
    "absolutely", "always", "certainly", "clearly", "completely",
    "confident", "convinced", "decided", "definitely", "essential",
    "every", "everyone", "everything", "evident", "exactly",
    "forever", "fundamental", "guaranteed", "indeed", "inevitable",
    "must", "necessary", "never", "obvious", "obviously",
    "perfectly", "positive", "precisely", "surely", "total",
    "totally", "truth", "undeniable", "undeniably", "undoubtedly",
    "unequivocal", "unequivocally", "unquestionable", "unquestionably",
    "without question", "without doubt",
    # Commitment language
    "commit", "committed", "commitment", "promise", "pledge",
    "guarantee", "vow", "sworn", "dedicated", "determined",
    "resolute", "resolve", "firm", "firmly", "unwavering",
}

HEDGING_WORDS = {
    # Epistemic hedges
    "almost", "apparently", "appear", "appears", "arguably",
    "assume", "assumption", "believe", "broadly", "could",
    "fairly", "generally", "guess", "hopefully", "imagine",
    "indicate", "largely", "likely", "may", "maybe",
    "might", "mostly", "often", "perhaps", "plausible",
    "possibly", "potentially", "presumably", "probable", "probably",
    "quite", "rather", "relatively", "seem", "seems",
    "sometimes", "somewhat", "sort of", "suggest", "suppose",
    "tend", "typically", "uncertain", "unclear", "usually",
    # Qualifying phrases
    "in some ways", "to some extent", "more or less",
    "kind of", "i think", "i believe", "it seems",
    "as far as", "from my perspective",
}


# =============================================================================
# NEGATION (Newman et al., 2003)
# =============================================================================
# Increased negation correlates with deception and defensive positioning

NEGATION_WORDS = {
    "no", "not", "never", "neither", "nobody", "none", "nor",
    "nothing", "nowhere", "cannot", "can't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't",
}


# =============================================================================
# EXCLUSIVE WORDS (Newman et al., 2003)
# =============================================================================
# Low in deceptive speech - liars simplify, truth-tellers differentiate

EXCLUSIVE_WORDS = {
    "but", "except", "however", "nevertheless", "nonetheless",
    "although", "though", "whereas", "while", "unless",
    "without", "rather", "instead", "otherwise", "yet",
    "despite", "notwithstanding", "on the other hand",
    "alternatively", "conversely",
}


# =============================================================================
# CAUSATION / COGNITIVE PROCESS (Tausczik & Pennebaker, 2010)
# =============================================================================
# Causal reasoning indicates analytical processing of the topic

CAUSATION_WORDS = {
    "because", "cause", "caused", "causing", "consequently",
    "effect", "effects", "hence", "lead", "leads",
    "led", "produce", "produces", "reason", "reasons",
    "result", "resulted", "results", "since", "so",
    "therefore", "thus", "due to", "owing to", "as a result",
    "on account of", "by virtue of",
}

INSIGHT_WORDS = {
    "acknowledge", "appreciate", "aware", "conclude", "consider",
    "discover", "feel", "find", "found", "know",
    "knew", "learn", "learned", "notice", "noticed",
    "observe", "realize", "realized", "recognize", "see",
    "think", "thought", "understand", "understood",
}


# =============================================================================
# TEMPORAL ORIENTATION
# =============================================================================
# Shift from past/present to future signals commitment to action

FUTURE_WORDS = {
    "will", "shall", "going to", "gonna", "intend",
    "plan", "expect", "anticipate", "forecast", "predict",
    "upcoming", "tomorrow", "next", "soon", "eventually",
    "forward", "ahead", "future", "prospective", "forthcoming",
}

PAST_WORDS = {
    "was", "were", "had", "did", "said",
    "went", "came", "made", "took", "gave",
    "told", "found", "knew", "thought", "saw",
    "ago", "before", "earlier", "formerly", "once",
    "previously", "yesterday", "last", "prior", "past",
    "historical", "historically", "traditionally",
}

PRESENT_WORDS = {
    "is", "are", "am", "being", "do",
    "does", "has", "have", "currently", "now",
    "today", "presently", "ongoing", "existing", "active",
    "right now", "at this moment", "at present",
}


# =============================================================================
# EMOTION: ANGER (LIWC anger category approximation)
# =============================================================================

ANGER_WORDS = {
    "abuse", "abused", "anger", "angry", "annoy", "annoyed",
    "antagonize", "arrogant", "assault", "attack", "bastard",
    "battle", "betray", "betrayed", "bitter", "blame",
    "bother", "bully", "cheat", "complain", "confront",
    "contempt", "corrupt", "cruel", "crush", "damage",
    "dangerous", "deceive", "defy", "demand", "demolish",
    "despise", "destroy", "destruction", "detest", "disgust",
    "disrespect", "dominate", "enemy", "enrage", "exploit",
    "fight", "force", "fraud", "frustrate", "furious",
    "greed", "hate", "hatred", "hostile", "humiliate",
    "hurt", "insult", "intimidate", "jealous", "kill",
    "lie", "lied", "liar", "manipulate", "mock",
    "offend", "oppose", "outrage", "outrageous", "punish",
    "rage", "reject", "resent", "revenge", "ridicule",
    "ruin", "sabotage", "shame", "steal", "stubborn",
    "stupid", "terrible", "terrify", "threat", "threaten",
    "toxic", "ugly", "unfair", "vicious", "victim",
    "violate", "violent", "war", "weapon", "wicked", "wrong",
}


# =============================================================================
# EMOTION: ANXIETY/FEAR
# =============================================================================

ANXIETY_WORDS = {
    "afraid", "alarm", "alarmed", "anxious", "anxiety",
    "apprehension", "avoid", "caution", "concern", "concerned",
    "confuse", "confused", "crisis", "danger", "dread",
    "doubt", "emergency", "expose", "fear", "fearful",
    "fright", "frighten", "hesitant", "hesitate", "horror",
    "insecure", "nervous", "overwhelm", "panic", "paranoid",
    "peril", "risk", "risky", "scare", "scared",
    "shock", "stress", "tense", "tension", "terrified",
    "terror", "threat", "trouble", "uncertain", "uneasy",
    "unpredictable", "unsafe", "unstable", "vulnerable",
    "warning", "worry", "worried",
}


# =============================================================================
# EMOTION: POSITIVE
# =============================================================================

POSITIVE_WORDS = {
    "accomplish", "achieve", "achievement", "admire", "advantage",
    "agree", "amazing", "appreciate", "approval", "approve",
    "beautiful", "benefit", "best", "better", "bless",
    "blessing", "brave", "brilliant", "celebrate", "champion",
    "cheerful", "compassion", "confidence", "courage", "delight",
    "effective", "empower", "encourage", "enjoy", "enthusiasm",
    "excellent", "excited", "exciting", "extraordinary", "fair",
    "faith", "fantastic", "freedom", "generous", "glad",
    "glory", "good", "grace", "grateful", "great",
    "grow", "growth", "happy", "harmony", "heal",
    "hero", "heroic", "honor", "hope", "hopeful",
    "improve", "incredible", "inspire", "inspiring", "integrity",
    "joy", "justice", "kind", "kindness", "lead",
    "liberty", "love", "miracle", "moral", "noble",
    "opportunity", "optimistic", "outstanding", "passion", "peace",
    "perfect", "positive", "power", "powerful", "pride",
    "progress", "promise", "prosper", "prosperity", "proud",
    "remarkable", "respect", "restore", "safe", "secure",
    "solution", "strength", "strong", "succeed", "success",
    "support", "thrive", "together", "triumph", "trust",
    "truth", "unite", "united", "unity", "valuable",
    "victory", "virtue", "vision", "win", "wisdom",
    "wonderful",
}


# =============================================================================
# POWER / DOMINANCE (Kacewicz et al., 2014)
# =============================================================================
# High-status speakers use more we-words, fewer I-words, more certainty

POWER_WORDS = {
    "authority", "authorize", "boss", "command", "compel",
    "control", "decide", "decision", "decree", "demand",
    "direct", "directive", "dominate", "dominion", "enforce",
    "execute", "force", "govern", "impose", "influence",
    "instruct", "lead", "leadership", "mandate", "master",
    "order", "override", "overrule", "power", "powerful",
    "preside", "pressure", "prohibit", "regulate", "reign",
    "require", "restrict", "rule", "sanction", "superior",
    "supremacy", "veto",
}


# =============================================================================
# ACHIEVEMENT (LIWC category approximation)
# =============================================================================

ACHIEVEMENT_WORDS = {
    "accomplish", "accomplished", "accomplishment", "achieve", "achieved",
    "achievement", "attain", "award", "beat", "best",
    "better", "build", "built", "champion", "complete",
    "completed", "competitive", "create", "deliver", "delivered",
    "earn", "earned", "effective", "efficient", "effort",
    "excellence", "excel", "finish", "first", "gain",
    "goal", "goals", "grow", "growth", "improve",
    "improvement", "innovate", "innovation", "invest", "invested",
    "lead", "milestone", "outperform", "overcome", "perform",
    "performance", "produce", "productive", "productivity", "profit",
    "progress", "prosper", "rank", "record", "result",
    "results", "score", "solve", "solved", "succeed",
    "success", "successful", "surpass", "top", "win",
    "won", "yield",
}


# =============================================================================
# MONEY / ECONOMIC (for mapping to financial policy contracts)
# =============================================================================

ECONOMIC_WORDS = {
    "afford", "affordable", "bank", "banking", "billion",
    "bond", "bonds", "budget", "business", "capital",
    "commerce", "commercial", "consumer", "cost", "costs",
    "credit", "debt", "deficit", "dollar", "dollars",
    "earn", "earnings", "economic", "economy", "employment",
    "expense", "export", "finance", "financial", "fiscal",
    "fund", "funding", "gdp", "growth", "income",
    "industrial", "industry", "inflation", "interest",
    "invest", "investment", "investor", "job", "jobs",
    "labor", "loan", "manufacture", "manufacturing", "market",
    "markets", "million", "money", "mortgage", "pay",
    "payroll", "pension", "poverty", "price", "prices",
    "profit", "profits", "recession", "regulation", "revenue",
    "salary", "savings", "spending", "stock", "subsidy",
    "surplus", "tariff", "tariffs", "tax", "taxes",
    "trade", "trillion", "unemployment", "wage", "wages",
    "wall street", "wealth", "welfare",
}


# =============================================================================
# POLICY DOMAIN LEXICONS (for topic-signal mapping to prediction markets)
# =============================================================================

IMMIGRATION_WORDS = {
    "alien", "amnesty", "asylum", "border", "borders",
    "citizenship", "customs", "daca", "deport", "deportation",
    "detention", "dreamer", "dreamers", "emigrate", "foreign",
    "green card", "ice", "illegal", "immigrant", "immigrants",
    "immigration", "migrate", "migration", "naturalization",
    "refugee", "refugees", "sanctuary", "undocumented", "visa",
    "wall", "xenophobia",
}

HEALTHCARE_WORDS = {
    "aca", "affordable care", "clinic", "coverage", "diagnosis",
    "disease", "doctor", "doctors", "drug", "drugs",
    "epidemic", "fda", "health", "healthcare", "hospital",
    "hospitals", "insurance", "medicaid", "medical", "medicare",
    "medicine", "mental health", "nurse", "obamacare", "opioid",
    "pandemic", "patient", "patients", "pharmaceutical",
    "pharmacy", "physician", "prescription", "public health",
    "surgery", "treatment", "uninsured", "vaccine", "vaccines",
}

SECURITY_WORDS = {
    "armed forces", "army", "attack", "battlefield", "bomb",
    "cia", "classified", "combat", "counterterrorism", "cyber",
    "cybersecurity", "defend", "defense", "deploy", "deployment",
    "dhs", "drone", "espionage", "extremism", "fbi",
    "force", "homeland", "intelligence", "iran", "iraq",
    "isis", "marines", "military", "missile", "nato",
    "navy", "nuclear", "nsa", "pentagon", "security",
    "soldier", "soldiers", "surveillance", "taliban", "terror",
    "terrorism", "terrorist", "threat", "troops", "veteran",
    "veterans", "war", "warfare", "weapon", "weapons",
}

CLIMATE_WORDS = {
    "carbon", "clean energy", "climate", "coal", "conservation",
    "drought", "earthquake", "emission", "emissions", "endangered",
    "energy", "environment", "environmental", "epa", "flood",
    "fossil fuel", "fracking", "green", "greenhouse", "hurricane",
    "methane", "natural gas", "nuclear energy", "oil", "paris accord",
    "paris agreement", "pipeline", "pollution", "recycle", "renewable",
    "solar", "sustainability", "sustainable", "temperature", "warming",
    "water", "weather", "wildfire", "wind energy", "wind power",
}

GUN_WORDS = {
    "amendment", "ammunition", "ar-15", "arm", "armed",
    "arms", "assault weapon", "atf", "background check",
    "ban", "bullet", "caliber", "carry", "concealed",
    "constitutional", "firearm", "firearms", "gun", "guns",
    "handgun", "holster", "magazine", "militia", "nra",
    "pistol", "rifle", "second amendment", "semi-automatic",
    "shoot", "shooter", "shooting", "weapon",
}


# =============================================================================
# AGGREGATE DICTIONARY FOR EASY ACCESS
# =============================================================================

ALL_LEXICONS = {
    # Psychological dimensions
    "certainty": CERTAINTY_WORDS,
    "hedging": HEDGING_WORDS,
    "negation": NEGATION_WORDS,
    "exclusive": EXCLUSIVE_WORDS,
    "causation": CAUSATION_WORDS,
    "insight": INSIGHT_WORDS,
    "future": FUTURE_WORDS,
    "past": PAST_WORDS,
    "present": PRESENT_WORDS,
    "anger": ANGER_WORDS,
    "anxiety": ANXIETY_WORDS,
    "positive_emotion": POSITIVE_WORDS,
    "power": POWER_WORDS,
    "achievement": ACHIEVEMENT_WORDS,
    "economic": ECONOMIC_WORDS,
    # Policy domains
    "immigration": IMMIGRATION_WORDS,
    "healthcare": HEALTHCARE_WORDS,
    "security": SECURITY_WORDS,
    "climate": CLIMATE_WORDS,
    "guns": GUN_WORDS,
}

# Dimensions used for baseline profiling (not policy domains)
PROFILE_DIMENSIONS = [
    "certainty", "hedging", "negation", "exclusive", "causation",
    "insight", "future", "past", "present", "anger", "anxiety",
    "positive_emotion", "power", "achievement", "economic",
]

# Policy domains for topic mapping
POLICY_DOMAINS = [
    "immigration", "healthcare", "security", "climate", "guns", "economic",
]
