"""
Configuration file for Splink Organization Matching Model

Adjust these parameters to tune the model for your needs.
Higher thresholds = fewer false positives but may miss some matches
Lower thresholds = more matches but risk more false positives
"""

# ========================================
# MATCH THRESHOLD
# ========================================

# Primary match threshold for predictions
# Recommended: 0.85 (balanced), 0.90 (conservative), 0.80 (aggressive)
MATCH_THRESHOLD = 0.85

# Threshold for clustering
# Usually same as MATCH_THRESHOLD
CLUSTER_THRESHOLD = 0.85


# ========================================
# COMPARISON LEVELS - Name Matching
# ========================================

# Jaro-Winkler thresholds for name similarity
# Lower values = more lenient matching
# Higher values = stricter matching

# Very high confidence (almost identical names)
JARO_WINKLER_HIGH = 0.95

# High confidence
JARO_WINKLER_MEDIUM = 0.92

# Moderate confidence (use with caution)
JARO_WINKLER_LOW = 0.88

# Lowest threshold to consider
# Set to None to disable this level
JARO_WINKLER_VERY_LOW = 0.85

# Enable term frequency adjustments
# This increases weight for uncommon names
TERM_FREQUENCY_ADJUSTMENTS = True


# ========================================
# BLOCKING RULES
# ========================================

# Number of characters for exact prefix matching
# Higher = fewer comparisons, more performance, may miss some matches
# Lower = more comparisons, slower, catches more variations
PREFIX_MATCH_LENGTH_LONG = 15
PREFIX_MATCH_LENGTH_SHORT = 10

# Enable university name blocking
# If True, pairs must have same university to be compared
# Reduces false positives but may miss cross-university matches
REQUIRE_UNIVERSITY_MATCH_FOR_BLOCKING = False


# ========================================
# TRAINING SETTINGS
# ========================================

# Recall parameter for estimating probability two random records match
# Lower = more conservative prior
# Higher = more aggressive prior
PRIOR_RECALL = 0.7

# Blocking rules for EM training
# Use tighter rules for faster training
TRAINING_BLOCKING_PREFIX_LENGTH = 12


# ========================================
# EVALUATION SETTINGS
# ========================================

# Cluster quality thresholds
MIN_CLUSTER_DENSITY = 0.6  # Clusters below this are suspicious
MAX_CLUSTER_CENTRALIZATION = 0.7  # Clusters above this are suspicious

# Sample size for manual review
MANUAL_REVIEW_SAMPLE_SIZE = 30


# ========================================
# FALSE POSITIVE REDUCTION
# ========================================

# Enable post-processing filters
FILTER_LOW_DENSITY_CLUSTERS = True

# Require university name match in final results
# If True, remove matches where universities differ
REQUIRE_UNIVERSITY_MATCH_FINAL = False

# Minimum match weight (in addition to probability threshold)
# Set to None to disable
MIN_MATCH_WEIGHT = None


# ========================================
# PRESETS
# ========================================

def get_conservative_config():
    """
    Preset: Conservative (minimize false positives)
    Use when false positives are very costly
    """
    return {
        'MATCH_THRESHOLD': 0.92,
        'JARO_WINKLER_HIGH': 0.96,
        'JARO_WINKLER_MEDIUM': 0.93,
        'JARO_WINKLER_LOW': 0.90,
        'JARO_WINKLER_VERY_LOW': None,  # Disable
        'PREFIX_MATCH_LENGTH_LONG': 20,
        'REQUIRE_UNIVERSITY_MATCH_FINAL': True,
        'FILTER_LOW_DENSITY_CLUSTERS': True,
        'MIN_CLUSTER_DENSITY': 0.7
    }


def get_balanced_config():
    """
    Preset: Balanced (default)
    Use for general entity resolution
    """
    return {
        'MATCH_THRESHOLD': 0.85,
        'JARO_WINKLER_HIGH': 0.95,
        'JARO_WINKLER_MEDIUM': 0.92,
        'JARO_WINKLER_LOW': 0.88,
        'JARO_WINKLER_VERY_LOW': 0.85,
        'PREFIX_MATCH_LENGTH_LONG': 15,
        'REQUIRE_UNIVERSITY_MATCH_FINAL': False,
        'FILTER_LOW_DENSITY_CLUSTERS': True,
        'MIN_CLUSTER_DENSITY': 0.6
    }


def get_aggressive_config():
    """
    Preset: Aggressive (maximize recall)
    Use when you want to catch all possible matches
    You'll need to manually filter false positives
    """
    return {
        'MATCH_THRESHOLD': 0.75,
        'JARO_WINKLER_HIGH': 0.92,
        'JARO_WINKLER_MEDIUM': 0.88,
        'JARO_WINKLER_LOW': 0.85,
        'JARO_WINKLER_VERY_LOW': 0.80,
        'PREFIX_MATCH_LENGTH_LONG': 10,
        'REQUIRE_UNIVERSITY_MATCH_FINAL': False,
        'FILTER_LOW_DENSITY_CLUSTERS': False,
        'MIN_CLUSTER_DENSITY': 0.4
    }


def apply_preset(preset_name):
    """
    Apply a preset configuration

    Args:
        preset_name: 'conservative', 'balanced', or 'aggressive'
    """
    presets = {
        'conservative': get_conservative_config(),
        'balanced': get_balanced_config(),
        'aggressive': get_aggressive_config()
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from: {list(presets.keys())}")

    config = presets[preset_name]

    # Update global variables
    globals().update(config)

    return config


if __name__ == "__main__":
    print("Model Configuration")
    print("===================")
    print(f"Match Threshold: {MATCH_THRESHOLD}")
    print(f"Jaro-Winkler Levels: {JARO_WINKLER_HIGH}, {JARO_WINKLER_MEDIUM}, {JARO_WINKLER_LOW}")
    print(f"Prefix Match Length: {PREFIX_MATCH_LENGTH_LONG}")
    print(f"Filter Low Density Clusters: {FILTER_LOW_DENSITY_CLUSTERS}")
    print("\nAvailable presets:")
    print("  - conservative: Minimize false positives")
    print("  - balanced: Default settings")
    print("  - aggressive: Maximize recall")
    print("\nUsage:")
    print("  import model_config")
    print("  config = model_config.apply_preset('conservative')")
