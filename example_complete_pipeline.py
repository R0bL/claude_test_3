"""
Complete End-to-End Splink Pipeline Example
============================================

This example demonstrates a production-ready Splink pipeline using
advanced techniques to minimize false positives while maintaining recall.

Scenario: Matching hospital/research institution records with:
- Multiple name variations
- Geographic locations
- University affiliations
- Potential false positives from generic names

This combines all advanced techniques from the guide.
"""

import pandas as pd
import numpy as np
from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_level_library as cll

# Import our custom libraries
from advanced_comparison_library import (
    PhoneticComparisons,
    GeographicComparisons,
    DateComparisons,
    CompositeComparisons
)
from advanced_blocking_optimization import (
    BlockingRuleGenerator,
    SaltingStrategy,
    BlockingOptimizer,
    BlockingConfigs
)
from corporate_matching import CorporateEntityExtractor


# =============================================================================
# STEP 1: LOAD AND PREPARE DATA
# =============================================================================

def load_data():
    """
    Load your data here.
    For this example, we'll create synthetic data.
    """
    print("Loading data...")

    # Synthetic hospital data
    data = {
        'unique_id': range(1000),
        'name': [
            'First Affiliated Hospital of Beijing University',
            'First Affiliated Hospital Beijing University',  # Variation
            'First Affiliated Hospital of Shanghai University',  # Different institution!
            'Second Affiliated Hospital of Beijing University',
            'Beijing University Hospital',
            # ... more records
        ] * 200,  # Repeat to create 1000 records
        'city': ['Beijing', 'Shanghai', 'Guangzhou'] * 334,
        'founded_year': np.random.randint(1900, 2020, 1000),
        'latitude': np.random.uniform(30, 40, 1000),
        'longitude': np.random.uniform(110, 120, 1000),
    }

    df = pd.DataFrame(data)
    return df


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create derived features to improve matching.
    """
    print("\nEngineering features...")

    # Extract university name
    import re

    def extract_university(name):
        patterns = [
            r'of ([^,]+University)',
            r'Affiliated to ([^,]+University)',
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return match.group(1).strip()
        return None

    def extract_hospital_type(name):
        ordinal_match = re.search(
            r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)',
            name,
            re.IGNORECASE
        )
        if ordinal_match:
            return ordinal_match.group(1).lower()
        return None

    def normalize_name(name):
        name = name.lower().strip()
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name)
        return name

    df['university_name'] = df['name'].apply(extract_university)
    df['hospital_type'] = df['name'].apply(extract_hospital_type)
    df['name_normalized'] = df['name'].apply(normalize_name)

    # Add phonetic encoding
    df = PhoneticComparisons.prepare_phonetic_columns(df, 'name_normalized')

    # Add decade for fuzzy date matching
    df['founded_decade'] = (df['founded_year'] // 10) * 10

    print(f"  ✓ Added features: university_name, hospital_type, name_normalized, phonetic columns")
    print(f"  ✓ University names found: {df['university_name'].nunique()} unique")
    print(f"  ✓ Hospital types found: {df['hospital_type'].nunique()} unique")

    return df


# =============================================================================
# STEP 3: ANALYZE BLOCKING RULES
# =============================================================================

def analyze_blocking(df):
    """
    Analyze potential blocking rules before running the linkage.
    """
    print("\nAnalyzing blocking rules...")

    optimizer = BlockingOptimizer()

    candidate_blocking_rules = [
        "l.university_name = r.university_name",
        "l.city = r.city",
        "substr(l.name_normalized, 1, 20) = substr(r.name_normalized, 1, 20)",
    ]

    candidate_columns = ['university_name', 'city', 'name_normalized']

    analysis = optimizer.analyze_blocking_rule_coverage(
        df,
        candidate_blocking_rules,
        candidate_columns
    )

    print("\nBlocking Rule Analysis:")
    print(analysis.to_string())

    # Estimate salting needs
    salting = SaltingStrategy()

    for col in candidate_columns:
        if col in df.columns:
            partitions = salting.estimate_salting_partitions(df, col)
            print(f"  Salting recommendation for '{col}': {partitions} partitions")

    return analysis


# =============================================================================
# STEP 4: DEFINE SPLINK SETTINGS
# =============================================================================

def create_settings():
    """
    Create Splink settings with advanced comparisons.
    """
    print("\nCreating Splink settings...")

    settings = {
        "link_type": "dedupe_only",
        "unique_id_column_name": "unique_id",

        # Blocking rules (critical for false positive reduction)
        "blocking_rules_to_generate_predictions": [
            # Rule 1: Same university (CRITICAL - prevents FPs)
            "l.university_name = r.university_name AND l.university_name IS NOT NULL",

            # Rule 2: Same city + hospital type
            "l.city = r.city AND l.hospital_type = r.hospital_type",

            # Rule 3: Long name prefix (catches variations)
            "substr(l.name_normalized, 1, 25) = substr(r.name_normalized, 1, 25)",

            # Rule 4: Geographic proximity
            # (Custom SQL - within 10km)
            """
            6371 * acos(
                cos(radians(l.latitude)) *
                cos(radians(r.latitude)) *
                cos(radians(r.longitude) - radians(l.longitude)) +
                sin(radians(l.latitude)) *
                sin(radians(r.latitude))
            ) <= 10
            """,
        ],

        # Comparisons
        "comparisons": [
            # 1. Name comparison with phonetic
            {
                "output_column_name": "name",
                "comparison_levels": [
                    cll.NullLevel("name_normalized"),
                    cll.ExactMatchLevel("name_normalized", term_frequency_adjustments=True),
                    # Phonetic match
                    cll.ExactMatchLevel("name_normalized_dmeta"),
                    # Fuzzy string matching
                    cll.JaroWinklerLevel("name_normalized", 0.95, term_frequency_adjustments=True),
                    cll.JaroWinklerLevel("name_normalized", 0.90),
                    cll.ElseLevel()
                ]
            },

            # 2. University name (MUST MATCH for hospitals)
            {
                "output_column_name": "university",
                "comparison_levels": [
                    cll.NullLevel("university_name"),
                    cll.ExactMatchLevel("university_name"),
                    cll.JaroWinklerLevel("university_name", 0.95),
                    cll.ElseLevel()
                ]
            },

            # 3. City
            cl.ExactMatch("city"),

            # 4. Hospital type
            cl.ExactMatch("hospital_type"),

            # 5. Geographic distance
            {
                "output_column_name": "location",
                "comparison_levels": [
                    cll.NullLevel("latitude"),
                    # Within 1km (same location)
                    {
                        "sql_condition": """
                            6371 * acos(
                                cos(radians(l.latitude)) *
                                cos(radians(r.latitude)) *
                                cos(radians(r.longitude) - radians(l.longitude)) +
                                sin(radians(l.latitude)) *
                                sin(radians(r.latitude))
                            ) <= 1
                        """,
                        "label_for_charts": "Within 1km"
                    },
                    # Within 10km
                    {
                        "sql_condition": """
                            6371 * acos(
                                cos(radians(l.latitude)) *
                                cos(radians(r.latitude)) *
                                cos(radians(r.longitude) - radians(l.longitude)) +
                                sin(radians(l.latitude)) *
                                sin(radians(r.latitude))
                            ) <= 10
                        """,
                        "label_for_charts": "Within 10km"
                    },
                    cll.ElseLevel()
                ]
            },

            # 6. Founded year (fuzzy date)
            {
                "output_column_name": "founded_year",
                "comparison_levels": [
                    cll.NullLevel("founded_year"),
                    cll.ExactMatchLevel("founded_year"),
                    # Within 5 years (typos)
                    {
                        "sql_condition": "abs(l.founded_year - r.founded_year) <= 5",
                        "label_for_charts": "Within 5 years"
                    },
                    # Same decade
                    cll.ExactMatchLevel("founded_decade"),
                    cll.ElseLevel()
                ]
            },
        ],

        # Retain intermediate calculation columns for debugging
        "retain_matching_columns": True,
        "retain_intermediate_calculation_columns": True,

        # EM training settings
        "max_iterations": 20,
        "em_convergence": 0.001,
    }

    print("  ✓ Settings created with:")
    print(f"    - {len(settings['blocking_rules_to_generate_predictions'])} blocking rules")
    print(f"    - {len(settings['comparisons'])} comparison functions")
    print(f"    - Link type: {settings['link_type']}")

    return settings


# =============================================================================
# STEP 5: TRAIN MODEL
# =============================================================================

def train_model(linker):
    """
    Train Splink model using best practices.
    """
    print("\nTraining model...")

    # Step 1: Estimate u probabilities
    print("  1. Estimating u probabilities (random sampling)...")
    linker.estimate_u_using_random_sampling(max_pairs=1e7)

    # Step 2: Estimate probability two random records match
    print("  2. Estimating probability two random records match...")
    linker.estimate_probability_two_random_records_match(
        deterministic_matching_rules=[
            "l.name_normalized = r.name_normalized AND l.university_name = r.university_name",
        ],
        recall=0.7  # Conservative
    )

    # Step 3: EM training for m probabilities (multiple sessions)
    print("  3. EM training for m probabilities...")

    # Session 1: Block on university
    print("     Session 1: University blocking")
    session_1 = linker.estimate_m_from_blocking_rule(
        "l.university_name = r.university_name",
        fix_u_probabilities=True,
        fix_probability_two_random_records_match=True
    )

    # Session 2: Block on city
    print("     Session 2: City blocking")
    session_2 = linker.estimate_m_from_blocking_rule(
        "l.city = r.city",
        fix_u_probabilities=True,
        fix_probability_two_random_records_match=True
    )

    # Session 3: Block on name prefix
    print("     Session 3: Name prefix blocking")
    session_3 = linker.estimate_m_from_blocking_rule(
        "substr(l.name_normalized, 1, 20) = substr(r.name_normalized, 1, 20)",
        fix_u_probabilities=True,
        fix_probability_two_random_records_match=True
    )

    # Compare parameter estimates
    print("  4. Comparing parameter estimates across sessions...")
    linker.visualisations.parameter_estimate_comparisons_chart(
        [session_1, session_2, session_3],
        out_path="parameter_comparison.html"
    )
    print("     ✓ Saved to parameter_comparison.html")

    # Visualize m/u parameters
    linker.visualisations.m_u_parameters_chart(
        out_path="m_u_parameters.html"
    )
    print("     ✓ Saved to m_u_parameters.html")

    print("  ✓ Training complete")


# =============================================================================
# STEP 6: PREDICT AND CLUSTER
# =============================================================================

def predict_and_cluster(linker, threshold=0.85):
    """
    Generate predictions and cluster results.
    """
    print(f"\nGenerating predictions (threshold: {threshold})...")

    # Predict
    predictions = linker.predict(threshold_match_probability=threshold)

    print(f"  ✓ Found {len(predictions)} pairwise matches above threshold")

    # Cluster
    print("\nClustering predictions...")
    clusters = linker.cluster_pairwise_predictions_at_threshold(
        predictions,
        threshold_match_probability=threshold
    )

    print(f"  ✓ Generated {clusters['cluster_id'].nunique()} clusters")

    # Cluster statistics
    cluster_sizes = clusters.groupby('cluster_id').size()
    print(f"\n  Cluster size distribution:")
    print(f"    - Size 2: {(cluster_sizes == 2).sum()}")
    print(f"    - Size 3-5: {((cluster_sizes >= 3) & (cluster_sizes <= 5)).sum()}")
    print(f"    - Size 6-10: {((cluster_sizes >= 6) & (cluster_sizes <= 10)).sum()}")
    print(f"    - Size 10+: {(cluster_sizes > 10).sum()}")

    if (cluster_sizes > 10).any():
        print("\n  ⚠️  WARNING: Large clusters detected - review for false positives")

    return predictions, clusters


# =============================================================================
# STEP 7: EVALUATE RESULTS
# =============================================================================

def evaluate_results(linker, predictions, clusters, df):
    """
    Generate evaluation artifacts.
    """
    print("\nEvaluating results...")

    # 1. Waterfall chart (sample predictions)
    print("  1. Generating waterfall charts...")
    if len(predictions) > 0:
        sample_predictions = predictions.sample(min(5, len(predictions)))
        for idx, row in sample_predictions.iterrows():
            linker.visualisations.waterfall_chart(
                records=[row['unique_id_l'], row['unique_id_r']],
                out_path=f"waterfall_{idx}.html"
            )
        print(f"     ✓ Saved {len(sample_predictions)} waterfall charts")

    # 2. Match weights distribution
    print("  2. Generating match weights chart...")
    linker.visualisations.match_weights_chart(
        predictions,
        out_path="match_weights.html"
    )
    print("     ✓ Saved to match_weights.html")

    # 3. Cluster studio
    print("  3. Generating cluster studio dashboard...")
    try:
        from splink import cluster_studio
        cluster_studio.create_cluster_studio_dashboard(
            predictions_df=predictions,
            clusters_df=clusters,
            out_path="cluster_studio.html",
            sample_size=100
        )
        print("     ✓ Saved to cluster_studio.html")
    except ImportError:
        print("     ⚠️  cluster_studio not available (install splink_cluster_studio)")

    # 4. False positive analysis
    print("  4. Analyzing potential false positives...")
    false_positive_candidates = identify_false_positive_candidates(predictions, df)
    print(f"     Found {len(false_positive_candidates)} potential false positives")
    print("\n     Top 5 suspicious matches:")
    print(false_positive_candidates.head(5)[['name_l', 'name_r', 'university_name_l', 'university_name_r', 'match_probability']])

    print("\n  ✓ Evaluation complete")


def identify_false_positive_candidates(predictions, df):
    """
    Identify potential false positives using domain knowledge.
    """
    # Merge with original data
    predictions_with_features = predictions.merge(
        df[['unique_id', 'name', 'university_name']],
        left_on='unique_id_l',
        right_on='unique_id',
        how='left'
    ).merge(
        df[['unique_id', 'name', 'university_name']],
        left_on='unique_id_r',
        right_on='unique_id',
        how='left',
        suffixes=('_l', '_r')
    )

    # Flag: High similarity name but DIFFERENT university
    predictions_with_features['potential_fp'] = (
        (predictions_with_features['university_name_l'] != predictions_with_features['university_name_r']) &
        (predictions_with_features['university_name_l'].notna()) &
        (predictions_with_features['university_name_r'].notna())
    )

    return predictions_with_features[predictions_with_features['potential_fp']].sort_values(
        'match_probability', ascending=False
    )


# =============================================================================
# STEP 8: POST-PROCESSING (ADDITIONAL FILTERING)
# =============================================================================

def post_process_matches(predictions, df):
    """
    Apply additional business rules to filter false positives.
    """
    print("\nPost-processing matches...")

    initial_count = len(predictions)

    # Merge with features
    predictions_enhanced = predictions.merge(
        df[['unique_id', 'university_name', 'hospital_type']],
        left_on='unique_id_l',
        right_on='unique_id',
        how='left'
    ).merge(
        df[['unique_id', 'university_name', 'hospital_type']],
        left_on='unique_id_r',
        right_on='unique_id',
        how='left',
        suffixes=('_l', '_r')
    )

    # Rule 1: MUST have same university (if both have university)
    predictions_enhanced = predictions_enhanced[
        ~(
            (predictions_enhanced['university_name_l'] != predictions_enhanced['university_name_r']) &
            (predictions_enhanced['university_name_l'].notna()) &
            (predictions_enhanced['university_name_r'].notna())
        )
    ]

    removed = initial_count - len(predictions_enhanced)
    print(f"  ✓ Removed {removed} matches with different universities")

    # Rule 2: Remove matches between different hospital types (First vs Second)
    # unless match probability is very high
    suspicious = (
        (predictions_enhanced['hospital_type_l'] != predictions_enhanced['hospital_type_r']) &
        (predictions_enhanced['hospital_type_l'].notna()) &
        (predictions_enhanced['hospital_type_r'].notna()) &
        (predictions_enhanced['match_probability'] < 0.95)
    )

    predictions_enhanced = predictions_enhanced[~suspicious]

    removed2 = len(predictions_enhanced)
    print(f"  ✓ Removed {initial_count - removed - removed2} matches between different hospital types")

    print(f"  ✓ Total removed: {initial_count - len(predictions_enhanced)}")
    print(f"  ✓ Remaining matches: {len(predictions_enhanced)}")

    return predictions_enhanced


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Run complete pipeline.
    """
    print("=" * 70)
    print("SPLINK ADVANCED TECHNIQUES - COMPLETE PIPELINE")
    print("=" * 70)

    # 1. Load data
    df = load_data()
    print(f"\n✓ Loaded {len(df)} records")

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Analyze blocking
    blocking_analysis = analyze_blocking(df)

    # 4. Create settings
    settings = create_settings()

    # 5. Initialize linker
    print("\nInitializing Splink linker...")
    linker = DuckDBLinker(
        df,
        settings,
        connection=":memory:"  # Or ":temp:" for on-disk
    )
    print("  ✓ Linker initialized (DuckDB backend)")

    # 6. Train model
    train_model(linker)

    # 7. Predict and cluster
    threshold = 0.85
    predictions, clusters = predict_and_cluster(linker, threshold)

    # 8. Evaluate
    evaluate_results(linker, predictions, clusters, df)

    # 9. Post-process
    predictions_filtered = post_process_matches(predictions, df)

    # 10. Save results
    print("\nSaving results...")
    predictions_filtered.to_csv("predictions_final.csv", index=False)
    clusters.to_csv("clusters.csv", index=False)
    print("  ✓ Saved to predictions_final.csv and clusters.csv")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  - Input records: {len(df)}")
    print(f"  - Matches found: {len(predictions_filtered)}")
    print(f"  - Clusters: {clusters['cluster_id'].nunique()}")
    print(f"  - Singleton records: {len(df) - len(clusters)}")
    print(f"\nNext steps:")
    print(f"  1. Review cluster_studio.html for manual validation")
    print(f"  2. Check waterfall charts to understand predictions")
    print(f"  3. Review parameter_comparison.html for training consistency")
    print(f"  4. Create labeled test set for precision/recall evaluation")


if __name__ == "__main__":
    main()
