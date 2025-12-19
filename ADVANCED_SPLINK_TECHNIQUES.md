# Advanced Splink Techniques for Entity Resolution

**A Comprehensive Guide to Production-Ready Data Linkage**

This guide covers advanced techniques for using Splink to build high-quality entity resolution systems that minimize false positives while maintaining strong recall.

---

## Table of Contents

1. [Advanced Comparison Functions](#1-advanced-comparison-functions)
2. [Blocking Optimization](#2-blocking-optimization)
3. [Model Training Improvements](#3-model-training-improvements)
4. [Evaluation and Diagnostics](#4-evaluation-and-diagnostics)
5. [Real-World Patterns](#5-real-world-patterns)
6. [Performance Optimization](#6-performance-optimization)
7. [Advanced Features](#7-advanced-features)

---

## 1. Advanced Comparison Functions

### 1.1 Phonetic Matching

Splink supports phonetic algorithms to match names that sound similar but are spelled differently.

#### Double Metaphone (Recommended)

```python
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_level_library as cll

# Feature engineering: Create phonetic columns
from phonetics import dmetaphone

df['name_dmeta'] = df['name'].apply(lambda x: dmetaphone(x)[0])

# Use in comparison
name_comparison = cl.NameComparison(
    "name",
    dmeta_col_name="name_dmeta",  # Uses Double Metaphone column
    term_frequency_adjustments=True
)
```

#### Custom Phonetic Comparisons

```python
# Using Soundex or Metaphone via feature engineering
from phonetics import soundex, metaphone

df['name_soundex'] = df['name'].apply(soundex)
df['name_metaphone'] = df['name'].apply(metaphone)

# Then compare using exact match on phonetic columns
phonetic_comparison = cl.CustomComparison(
    output_column_name="name_phonetic",
    comparison_levels=[
        cll.NullLevel("name"),
        cll.ExactMatchLevel("name"),  # Exact name match
        cll.ExactMatchLevel("name_dmeta"),  # Same double metaphone
        cll.ExactMatchLevel("name_soundex"),  # Same soundex
        cll.JaroWinklerLevel("name", 0.95),  # High similarity
        cll.ElseLevel()
    ]
)
```

**Key Insight**: Phonetic matching is particularly useful for names and can be used as additional comparison levels within name comparisons. Double Metaphone is preferred over Soundex for English names.

### 1.2 Token-Based Comparisons (Jaccard, Cosine)

#### Jaccard Similarity for Multi-Word Strings

```python
# Jaccard similarity for addresses or organization names
address_comparison = cl.CustomComparison(
    output_column_name="address",
    comparison_levels=[
        cll.NullLevel("address"),
        cll.ExactMatchLevel("address"),
        cll.JaccardLevel("address", 0.9),  # 90% token overlap
        cll.JaccardLevel("address", 0.7),  # 70% token overlap
        cll.JaccardLevel("address", 0.5),  # 50% token overlap
        cll.ElseLevel()
    ]
)
```

**Important Note**: Jaccard in Splink works best with strings that can be split into multiple words (e.g., addresses) rather than character-level comparisons. Token-based functionality for splitting strings into words is available via `jaccard_level()` and `jaccard_at_thresholds()`.

#### Cosine Similarity (Custom Implementation)

```python
# Cosine similarity is not built-in but can be added via custom SQL
# Currently, there's a feature request for embedding-based cosine similarity

# Workaround: Pre-compute embeddings and use custom distance function
from splink.comparison_level import ComparisonLevel

custom_cosine_level = ComparisonLevel({
    "sql_condition": """
        array_cosine_similarity(l.name_embedding, r.name_embedding) >= 0.85
    """,
    "label_for_charts": "Cosine similarity >= 0.85"
})
```

**Status**: As of 2025, Cosine similarity for embeddings is requested but not yet built-in. Users can implement custom SQL when the backend supports it.

### 1.3 Array/List Comparisons for Multiple Names/Aliases

```python
# Exact matching on array-valued columns (e.g., aliases)
alias_comparison = cl.ArrayIntersectComparison(
    "aliases",
    size_threshold_or_sizes=[3, 1],  # Match if 3+ common OR any 1 match
    term_frequency_adjustments=False
)

# Example with specific sizes
alias_comparison_exact = cl.CustomComparison(
    output_column_name="aliases",
    comparison_levels=[
        cll.NullLevel("aliases"),
        cll.ArrayIntersectLevel("aliases", min_intersection=3),
        cll.ArrayIntersectLevel("aliases", min_intersection=1),
        cll.ElseLevel()
    ]
)
```

**Current Limitation**: Array comparisons currently require exact matches among elements. Fuzzy matching within arrays is a requested feature (#1994).

**Workaround for Fuzzy Array Matching**:

```python
# Option 1: Explode arrays and use multiple comparisons
# Preprocess: Create first_alias, second_alias, third_alias columns

# Option 2: Use custom SQL with ANY/ALL clauses
from splink.comparison_level import ComparisonLevel

fuzzy_array_level = ComparisonLevel({
    "sql_condition": """
        EXISTS (
            SELECT 1 FROM unnest(l.aliases) AS la
            CROSS JOIN unnest(r.aliases) AS ra
            WHERE jaro_winkler_similarity(la, ra) >= 0.9
        )
    """,
    "label_for_charts": "Any alias matches fuzzy"
})
```

### 1.4 Fuzzy Date Matching

```python
# Date of Birth comparison with fuzzy matching
dob_comparison = cl.DateOfBirthComparison(
    "date_of_birth",
    input_is_string=True,  # If dates are strings
    datetime_format="%Y-%m-%d",
    datetime_thresholds=[1, 12, 120],  # days, months (1 year), months (10 years)
    term_frequency_adjustments=True
)

# More detailed custom date comparison
custom_date_comparison = cl.CustomComparison(
    output_column_name="event_date",
    comparison_levels=[
        cll.NullLevel("event_date"),
        cll.ExactMatchLevel("event_date"),
        # Dates within 1 day (typos)
        cll.DatediffLevel(
            "event_date",
            date_threshold=1,
            date_metric="day"
        ),
        # Dates within 1 month
        cll.DatediffLevel(
            "event_date",
            date_threshold=1,
            date_metric="month"
        ),
        # Dates within 1 year
        cll.DatediffLevel(
            "event_date",
            date_threshold=1,
            date_metric="year"
        ),
        # Same year only
        cll.ComparisonLevel({
            "sql_condition": "extract(year from l.event_date) = extract(year from r.event_date)",
            "label_for_charts": "Same year"
        }),
        cll.ElseLevel()
    ]
)
```

**Key Features**:
- Exact match on date
- String similarity (e.g., Damerau-Levenshtein ≤ 1 for typos)
- Temporal metrics: "year" and "month" with customizable thresholds
- Absolute date differences (within 1 month or 1 year)

### 1.5 Geographic Distance Calculations

```python
# Haversine distance for lat/long coordinates
location_comparison = cl.CustomComparison(
    output_column_name="location",
    comparison_levels=[
        cll.NullLevel("lat"),  # Either null
        cll.ExactMatchLevel("lat", "long"),  # Exact same location
        # Within 1km
        cll.DistanceInKMLevel("lat", "long", km_threshold=1),
        # Within 10km
        cll.DistanceInKMLevel("lat", "long", km_threshold=10),
        # Within 50km
        cll.DistanceInKMLevel("lat", "long", km_threshold=50),
        # Within 100km
        cll.DistanceInKMLevel("lat", "long", km_threshold=100),
        cll.ElseLevel()
    ]
)

# Using with PostcodeComparison
postcode_comparison = cl.PostcodeComparison(
    "postcode",
    lat_col="latitude",
    long_col="longitude",
    km_thresholds=[1, 10, 50]  # Distance thresholds in km
)
```

**Built-in Support**: Splink has built-in support for the haversine formula to transform lat/lng comparisons into distances measured in kilometers.

---

## 2. Blocking Optimization

### 2.1 Salting Techniques for Load Balancing

Salting helps distribute workload evenly across partitions, especially important for Spark backends.

```python
from splink.duckdb.linker import DuckDBLinker

linker = DuckDBLinker(
    df,
    settings={
        "link_type": "dedupe_only",
        "blocking_rules_to_generate_predictions": [
            # Simple blocking rule without salting
            "l.surname = r.surname",

            # Salted blocking rule for load balancing
            {
                "blocking_rule": "l.first_name = r.first_name",
                "salting_partitions": 4  # Split into 4 partitions
            },

            # Heavy blocking rule with more salting
            {
                "blocking_rule": "l.postcode = r.postcode",
                "salting_partitions": 10  # More partitions for skewed data
            }
        ],
        "comparisons": [...]
    }
)
```

**When to Use Salting**:
- Very large linkages (100M+ records)
- Skewed data (e.g., many "John Smith" vs few "Xerxes Zenobia")
- Spark backend (DuckDB handles parallelism automatically from v3.9.11+)
- Out of memory errors

**Performance Impact**:
- Users report 10-15 minute improvements with optimized salting
- Formula: `total_parallelism = base_parallelism × total_salted_partitions`
- Example: Block 1 with 120 partitions, Block 2 with 20, Block 3 with 80

### 2.2 Multi-Column Blocking Rules

```python
blocking_rules = [
    # Block 1: Exact match on single field
    "l.email = r.email",

    # Block 2: Multi-column exact match
    "l.first_name = r.first_name AND l.surname = r.surname",

    # Block 3: Combination with date
    "l.postcode = r.postcode AND l.dob = r.dob",

    # Block 4: Prefix + exact match
    "substr(l.company_name, 1, 10) = substr(r.company_name, 1, 10) AND l.country = r.country",

    # Block 5: OR conditions (use separate rules)
    "l.phone = r.phone",  # Rule A
    "l.email = r.email",  # Rule B (records blocked if either matches)

    # Block 6: Complex SQL logic
    """
    (l.surname = r.surname AND levenshtein(l.first_name, r.first_name) <= 2)
    OR
    (l.email_domain = r.email_domain AND l.employee_id = r.employee_id)
    """
]
```

### 2.3 Performance Optimization for Large Datasets

**Blocking Rule Strategy**:

```python
# TRAINING: Use tight blocking for speed
training_blocking_rules = [
    "l.surname = r.surname AND l.first_name = r.first_name",  # Tight
]

linker.estimate_u_using_random_sampling(max_pairs=1e7)
linker.estimate_m_from_pairwise_labels(training_blocking_rules)

# PREDICTION: Use broader blocking to catch all matches
prediction_blocking_rules = [
    "l.surname = r.surname",  # Broader
    "l.email = r.email",
    "substr(l.surname, 1, 3) = substr(r.surname, 1, 3) AND l.dob = r.dob",
]

predictions = linker.predict(threshold_match_probability=0.85)
```

**Key Insight**: Training blocking rules don't need to capture all true matches—just need to generate examples. Prediction blocking rules should be comprehensive.

### 2.4 Best Practices for Blocking on Arrays/Lists

```python
# Strategy 1: Create derived columns
df['primary_alias'] = df['aliases'].apply(lambda x: x[0] if x else None)
df['alias_count'] = df['aliases'].apply(len)

blocking_rules = [
    # Block on primary alias
    "l.primary_alias = r.primary_alias",

    # Block where both have multiple aliases
    "l.alias_count >= 2 AND r.alias_count >= 2 AND l.primary_alias = r.primary_alias",
]

# Strategy 2: Use array_intersect in blocking
blocking_rules = [
    # At least one alias in common
    "array_length(array_intersect(l.aliases, r.aliases)) >= 1"
]
```

---

## 3. Model Training Improvements

### 3.1 EM Training Convergence Best Practices

```python
from splink.duckdb.linker import DuckDBLinker

linker = DuckDBLinker(df, settings)

# Step 1: Estimate u probabilities (anchors the training)
linker.estimate_u_using_random_sampling(max_pairs=1e7)

# Step 2: Estimate probability two random records match
# This is critical for convergence
linker.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        "l.email = r.email",
        "l.phone = r.phone AND l.surname = r.surname"
    ],
    recall=0.7  # Conservative: assume rules catch 70% of matches
)

# Step 3: EM training for m probabilities
# Use multiple blocking rules for robustness
session_1 = linker.estimate_m_from_blocking_rule(
    blocking_rule="l.surname = r.surname",
    fix_u_probabilities=True,  # Keep u fixed (recommended)
    fix_probability_two_random_records_match=True  # Keep lambda fixed
)

session_2 = linker.estimate_m_from_blocking_rule(
    blocking_rule="l.postcode = r.postcode",
    fix_u_probabilities=True,
    fix_probability_two_random_records_match=True
)

session_3 = linker.estimate_m_from_blocking_rule(
    blocking_rule="l.dob = r.dob",
    fix_u_probabilities=True,
    fix_probability_two_random_records_match=True
)

# Step 4: Compare parameter estimates for consistency
linker.visualisations.parameter_estimate_comparisons_chart([
    session_1, session_2, session_3
])
```

**Why This Works**:
- **Anchoring**: Fixed u and λ parameters prevent EM from converging to local maxima
- **Multiple sessions**: Different blocking rules provide different sample populations
- **Averaging**: Splink averages estimates across sessions for robustness

**Convergence Indicators**:
- Parameter estimates stabilize across iterations
- Estimates from different blocking rules are similar
- Match weights make intuitive sense

### 3.2 Using Multiple Training Sessions on Different Blocking Rules

```python
# Pattern: Train on diverse blocking rules to get robust estimates

training_sessions = []

# Session 1: Geographic blocking
sessions_1 = linker.estimate_m_from_blocking_rule(
    "l.city = r.city",
    fix_u_probabilities=True
)
training_sessions.append(sessions_1)

# Session 2: Name blocking
sessions_2 = linker.estimate_m_from_blocking_rule(
    "l.surname = r.surname",
    fix_u_probabilities=True
)
training_sessions.append(sessions_2)

# Session 3: Identifier blocking
sessions_3 = linker.estimate_m_from_blocking_rule(
    "l.customer_id = r.customer_id",
    fix_u_probabilities=True
)
training_sessions.append(sessions_3)

# Compare estimates
linker.visualisations.parameter_estimate_comparisons_chart(training_sessions)

# If estimates are very different, investigate:
# - Are blocking rules introducing bias?
# - Is one blocking rule too restrictive?
# - Do you need separate models for different data segments?
```

**Best Practice**: Use 2-4 different blocking rules that capture different types of matches.

### 3.3 Handling Class Imbalance

#### Term Frequency Adjustments

```python
# Automatically handles rare vs common values
name_comparison = cl.NameComparison(
    "name",
    term_frequency_adjustments=True  # Upweight rare names, downweight common
)

# For very rare values, set a minimum u value
name_comparison = cl.CustomComparison(
    output_column_name="name",
    comparison_levels=[
        cll.NullLevel("name"),
        cll.ExactMatchLevel("name", term_frequency_adjustments=True,
                           tf_minimum_u_value=0.001),  # Prevent extreme weights
        cll.JaroWinklerLevel("name", 0.95, term_frequency_adjustments=True,
                            tf_minimum_u_value=0.001),
        cll.ElseLevel()
    ]
)
```

**How It Works**:
- Match on "Portia" → increased match weight (rare name)
- Match on "Jack" → decreased match weight (common name)
- `tf_minimum_u_value`: Prevents rare misspellings from getting disproportionate weight

#### Handling Very Common Names

```python
# Example: Organizations with generic names like "First Affiliated Hospital"

# Strategy 1: Require additional evidence
org_comparison = cl.CustomComparison(
    output_column_name="organization",
    comparison_levels=[
        cll.NullLevel("org_name"),
        # Exact match + same location
        cll.and_(
            cll.ExactMatchLevel("org_name"),
            cll.ExactMatchLevel("city")
        ),
        # Fuzzy match + same location
        cll.and_(
            cll.JaroWinklerLevel("org_name", 0.95),
            cll.ExactMatchLevel("city")
        ),
        cll.ElseLevel()
    ]
)

# Strategy 2: Use term frequency adjustments
# (automatically handles this - common names get lower weight)

# Strategy 3: Stricter blocking
blocking_rules = [
    # Don't just block on org name - add disambiguating field
    "l.org_name = r.org_name AND l.state = r.state"
]
```

### 3.4 When to Use Labeled vs Unsupervised Training

**Splink's Hybrid Approach (Recommended)**:

```python
# 1. Direct estimation (no labels needed)
linker.estimate_u_using_random_sampling(max_pairs=1e7)
linker.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[...],
    recall=0.7
)

# 2. Unsupervised EM for m probabilities
linker.estimate_m_from_blocking_rule("l.surname = r.surname")

# 3. Optional: Use labels for validation/threshold selection
labels_df = pd.DataFrame([
    {"unique_id_l": "id1", "unique_id_r": "id2", "clerical_match_score": 1},
    {"unique_id_l": "id3", "unique_id_r": "id4", "clerical_match_score": 0},
])

# Evaluate with labels
linker.visualisations.threshold_selection_tool_from_labels_table(
    labels_df,
    out_path="threshold_chart.html"
)
```

**When to Use Labeled Data**:
- ✅ **Evaluation**: Always use labels to validate model quality
- ✅ **Threshold selection**: Optimize precision/recall trade-off
- ✅ **Edge cases**: Create labels for specific problem patterns
- ❌ **Training all parameters**: EM often performs as well without labels
- ❌ **When labels are expensive**: Unsupervised works well for most cases

**Research Finding** (2025): "The gains in accuracy from more complex models are similar between supervised and unsupervised approaches."

---

## 4. Evaluation and Diagnostics

### 4.1 Interactive Charts for Model Diagnostics

#### Waterfall Chart

```python
# Understand how individual predictions are computed
linker.visualisations.waterfall_chart(
    records=[id1, id2],  # Two specific records
    filter_nulls=True,
    out_path="waterfall.html"
)
```

**How to Read**:
- First bar: Prior (baseline match weight before evidence)
- Subsequent bars: Contribution of each comparison
- Positive bars: Evidence FOR match
- Negative bars: Evidence AGAINST match
- Final bar: Total match weight → probability

**Diagnostic Use**:
- If prior is too large/small compared to comparison weights, retune `probability_two_random_records_match`
- If unexpected features dominate, check feature engineering
- If match weights seem wrong, revisit EM training

#### Match Weights Chart

```python
# Distribution of match weights across all comparisons
linker.visualisations.match_weights_chart(
    predictions_df,
    out_path="match_weights.html"
)
```

Shows histogram of partial match weights for each comparison level.

#### M/U Parameters Chart

```python
# Visualize trained m and u probabilities
linker.visualisations.m_u_parameters_chart(
    out_path="m_u_chart.html"
)
```

**What to Look For**:
- m probabilities should be high (matches agree on this field)
- u probabilities should be low (random records rarely agree)
- Large gap between m and u = strong discriminating feature

#### Parameter Estimate Comparisons

```python
# Compare estimates from different training sessions
linker.visualisations.parameter_estimate_comparisons_chart(
    [session_1, session_2, session_3],
    out_path="param_comparison.html"
)
```

**Red Flag**: If estimates differ significantly, investigate blocking rule bias.

### 4.2 Cluster Studio for Manual Review

```python
from splink import cluster_studio

# Generate clusters
clusters = linker.cluster_pairwise_predictions_at_threshold(
    predictions_df,
    threshold_match_probability=0.85
)

# Create interactive dashboard
cluster_studio.create_cluster_studio_dashboard(
    predictions_df=predictions_df,
    clusters_df=clusters,
    out_path="cluster_studio.html",
    sampling_method="random",  # or "by_cluster_size"
    sample_size=100
)
```

**Features**:
- Visualize individual clusters as graphs
- See all records in each cluster
- Inspect edge weights (match probabilities)
- Identify false positive/negative links

**Diagnostic Patterns**:
- **Star pattern** (high centralization): One record linked to many → likely false positives
- **Low density**: Sparse connections → suspicious cluster
- **Very large clusters**: May indicate chain of false positives

### 4.3 Precision-Recall Trade-offs

```python
# Interactive threshold selection tool
linker.visualisations.threshold_selection_tool_from_labels_table(
    labels_df,
    out_path="threshold_tool.html"
)
```

**Shows**:
- Precision and recall at different thresholds
- F₁, F₂, F₀.₅ scores
- P₄ score (addresses F-score limitations)
- True positive / false positive counts

**How to Choose Threshold**:

| Use Case | Optimize For | Typical Threshold |
|----------|-------------|-------------------|
| High-stakes matching (e.g., medical records) | Precision | 0.90 - 0.95 |
| General deduplication | F₁ score | 0.80 - 0.90 |
| Candidate generation for manual review | Recall | 0.70 - 0.80 |

### 4.4 False Positive/Negative Analysis

```python
# Create labels table
labels_df = pd.DataFrame([
    {"unique_id_l": "A", "unique_id_r": "B", "clerical_match_score": 1},
    {"unique_id_l": "C", "unique_id_r": "D", "clerical_match_score": 0},
    # ... more labels
])

# Generate accuracy analysis
linker.visualisations.accuracy_analysis_from_labels_table(
    labels_df,
    output_type="threshold_selection",  # or "roc" or "table"
    out_path="accuracy.html"
)
```

**Analyzing False Positives**:

```python
# Find predictions above threshold that are labeled as non-matches
threshold = 0.85
predictions_with_labels = predictions_df.merge(labels_df, on=["unique_id_l", "unique_id_r"])

false_positives = predictions_with_labels[
    (predictions_with_labels["match_probability"] >= threshold) &
    (predictions_with_labels["clerical_match_score"] == 0)
]

print(f"False positives: {len(false_positives)}")
print("\nCommon patterns:")
print(false_positives[["name_l", "name_r", "match_probability"]].head(10))
```

**Analyzing False Negatives**:

```python
false_negatives = predictions_with_labels[
    (predictions_with_labels["match_probability"] < threshold) &
    (predictions_with_labels["clerical_match_score"] == 1)
]

print(f"False negatives: {len(false_negatives)}")

# Check if blocking rules are too tight
# (True matches that never get compared at all won't appear in predictions_df)
```

### 4.5 ROC Curves and Threshold Selection

```python
# ROC curve
linker.visualisations.accuracy_analysis_from_labels_table(
    labels_df,
    output_type="roc",
    out_path="roc_curve.html"
)

# Programmatic threshold selection
from sklearn.metrics import precision_recall_curve

y_true = predictions_with_labels["clerical_match_score"]
y_scores = predictions_with_labels["match_probability"]

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# Find threshold for 95% precision
target_precision = 0.95
idx = np.argmax(precision >= target_precision)
optimal_threshold = thresholds[idx]
print(f"Threshold for {target_precision} precision: {optimal_threshold}")

# Or optimize F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_f1_idx = np.argmax(f1_scores)
optimal_f1_threshold = thresholds[optimal_f1_idx]
print(f"Threshold for optimal F1: {optimal_f1_threshold}")
```

---

## 5. Real-World Patterns

### 5.1 Multinational Corporations (Hierarchies, Subsidiaries)

**Pattern**: Companies with regional offices, subsidiaries, and complex ownership structures.

```python
# Feature engineering for corporate entities
import re

def extract_parent_company(name):
    """Extract parent from 'Subsidiary of Parent Inc'"""
    patterns = [
        r'subsidiary of (.+?)(?:,|$)',
        r'a (.+?) company',
        r'division of (.+?)(?:,|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return None

def extract_country_from_name(name):
    """Extract country from 'Company (Germany)'"""
    match = re.search(r'\(([A-Za-z\s]+)\)$', name)
    return match.group(1).strip() if match else None

def extract_core_company_name(name):
    """Remove suffixes, subsidiaries, addresses"""
    # Remove country codes
    name = re.sub(r'\s*\([A-Za-z\s]+\)\s*$', '', name)
    # Remove legal suffixes
    suffixes = ['Inc', 'LLC', 'Ltd', 'Corp', 'GmbH', 'AG', 'S.A.']
    for suffix in suffixes:
        name = re.sub(rf'\s+{suffix}\.?$', '', name, flags=re.IGNORECASE)
    return name.strip().lower()

# Apply feature engineering
df['parent_company'] = df['name'].apply(extract_parent_company)
df['country_from_name'] = df['name'].apply(extract_country_from_name)
df['core_company_name'] = df['name'].apply(extract_core_company_name)

# Comparison strategy
comparisons = [
    # Core company name
    cl.CustomComparison(
        output_column_name="core_company",
        comparison_levels=[
            cll.NullLevel("core_company_name"),
            cll.ExactMatchLevel("core_company_name"),
            cll.JaroWinklerLevel("core_company_name", 0.95),
            cll.JaroWinklerLevel("core_company_name", 0.90),
            cll.ElseLevel()
        ]
    ),
    # Parent company
    cl.CustomComparison(
        output_column_name="parent",
        comparison_levels=[
            cll.NullLevel("parent_company"),
            cll.ExactMatchLevel("parent_company"),
            cll.ElseLevel()
        ]
    ),
    # Country
    cl.ExactMatch("country_from_name"),
]

# Blocking rules
blocking_rules = [
    # Same core company name
    "l.core_company_name = r.core_company_name",
    # Same parent company
    "l.parent_company = r.parent_company AND l.parent_company IS NOT NULL",
    # Core name prefix + same country
    "substr(l.core_company_name, 1, 10) = substr(r.core_company_name, 1, 10) AND l.country_from_name = r.country_from_name",
]

# Post-processing: Decide match level
def determine_match_type(row):
    """
    Returns:
    - 'same_entity': Exact same entity
    - 'same_parent': Different subsidiaries, same parent
    - 'different': Different companies
    """
    if row['core_company_name_l'] == row['core_company_name_r']:
        if row['country_from_name_l'] == row['country_from_name_r']:
            return 'same_entity'
        elif row['country_from_name_l'] and row['country_from_name_r']:
            return 'same_parent'  # Different regional offices

    if row['parent_company_l'] and row['parent_company_r']:
        if row['parent_company_l'] == row['parent_company_r']:
            return 'same_parent'

    return 'different'

predictions_df['match_type'] = predictions_df.apply(determine_match_type, axis=1)

# Filter based on use case
if match_level == "entity":
    # Only exact entity matches
    predictions_df = predictions_df[predictions_df['match_type'] == 'same_entity']
elif match_level == "corporate_family":
    # Include same parent
    predictions_df = predictions_df[predictions_df['match_type'].isin(['same_entity', 'same_parent'])]
```

### 5.2 Research Institutions (Hospitals, Universities)

**Pattern**: "First Affiliated Hospital of X University" vs "First Affiliated Hospital of Y University"

```python
# Feature extraction
def extract_university(name):
    """Extract university name from hospital name"""
    patterns = [
        r'of ([^,]+University[^,]*)',
        r', ([^,]+University[^,]*)$',
        r'Affiliated to ([^,]+University[^,]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1).strip()
    return None

def extract_hospital_type(name):
    """Extract type: First/Second/Third, Children's, Cancer, etc."""
    # Ordinal numbers
    ordinal_match = re.search(r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth)', name, re.IGNORECASE)
    if ordinal_match:
        return ordinal_match.group(1).lower()

    # Specialty
    specialty_match = re.search(r"(Children's|Cancer|General|Teaching|Memorial|Central)", name, re.IGNORECASE)
    if specialty_match:
        return specialty_match.group(1).lower()

    return None

df['university_name'] = df['name'].apply(extract_university)
df['hospital_type'] = df['name'].apply(extract_hospital_type)

# Comparison requiring BOTH name similarity AND same university
from splink.comparison_level_composition import and_

hospital_comparison = cl.CustomComparison(
    output_column_name="hospital",
    comparison_levels=[
        cll.NullLevel("name"),
        # Exact name + same university
        and_(
            cll.ExactMatchLevel("name"),
            cll.ExactMatchLevel("university_name")
        ),
        # Fuzzy name + same university (reduces false positives!)
        and_(
            cll.JaroWinklerLevel("name", 0.95),
            cll.ExactMatchLevel("university_name")
        ),
        and_(
            cll.JaroWinklerLevel("name", 0.90),
            cll.ExactMatchLevel("university_name")
        ),
        cll.ElseLevel()
    ]
)

# Blocking rules
blocking_rules = [
    # Same university + hospital type
    "l.university_name = r.university_name AND l.hospital_type = r.hospital_type",
    # Same university only (broader)
    "l.university_name = r.university_name",
    # Name prefix
    "substr(l.name, 1, 20) = substr(r.name, 1, 20)",
]
```

**Critical**: Always require university match for hospitals to avoid false positives like:
- ❌ "First Affiliated Hospital of Zhengzhou University" ≠ "First Affiliated Hospital of Chongqing University"

### 5.3 Handling Multiple Name Variations and Aliases

```python
# Approach 1: Multiple name columns
df['official_name'] = ...
df['alias_1'] = ...
df['alias_2'] = ...
df['common_name'] = ...

# Compare all name fields
from splink.comparison_level_composition import or_

name_comparison = cl.CustomComparison(
    output_column_name="any_name",
    comparison_levels=[
        # Exact match on any name field
        or_(
            cll.ExactMatchLevel("official_name"),
            cll.ExactMatchLevel("alias_1"),
            cll.ExactMatchLevel("alias_2"),
            cll.ExactMatchLevel("common_name")
        ),
        # Fuzzy match on official name
        cll.JaroWinklerLevel("official_name", 0.95),
        cll.ElseLevel()
    ]
)

# Approach 2: Array of aliases
df['all_names'] = df.apply(
    lambda row: [row['official_name'], row['alias_1'], row['alias_2']],
    axis=1
)

# Custom SQL for fuzzy array matching
alias_fuzzy_level = cll.ComparisonLevel({
    "sql_condition": """
        EXISTS (
            SELECT 1
            FROM unnest(l.all_names) AS ln
            CROSS JOIN unnest(r.all_names) AS rn
            WHERE jaro_winkler_similarity(ln, rn) >= 0.95
        )
    """,
    "label_for_charts": "Any name fuzzy match >= 0.95"
})
```

### 5.4 Address Normalization Strategies

```python
import re

def normalize_address(address):
    """Normalize address for matching"""
    if pd.isna(address):
        return None

    address = address.lower().strip()

    # Remove punctuation
    address = re.sub(r'[.,]', ' ', address)

    # Standardize abbreviations
    replacements = {
        r'\bstreet\b': 'st',
        r'\bstraße\b': 'str',
        r'\broad\b': 'rd',
        r'\bavenue\b': 'ave',
        r'\bdrive\b': 'dr',
        r'\blane\b': 'ln',
        r'\bcourt\b': 'ct',
        r'\bapartment\b': 'apt',
        r'\bsuite\b': 'ste',
        r'\bnorth\b': 'n',
        r'\bsouth\b': 's',
        r'\beast\b': 'e',
        r'\bwest\b': 'w',
        r'\bfloor\b': 'fl',
        r'\bbuilding\b': 'bldg',
    }

    for pattern, replacement in replacements.items():
        address = re.sub(pattern, replacement, address)

    # Normalize whitespace
    address = re.sub(r'\s+', ' ', address)

    return address

def extract_postcode(address):
    """Extract postcode/ZIP"""
    # US ZIP
    us_zip = re.search(r'\b\d{5}(-\d{4})?\b', address)
    if us_zip:
        return us_zip.group(0)

    # UK postcode
    uk_postcode = re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', address, re.IGNORECASE)
    if uk_postcode:
        return uk_postcode.group(0).upper()

    return None

df['address_normalized'] = df['address'].apply(normalize_address)
df['postcode'] = df['address'].apply(extract_postcode)

# Comparison
address_comparison = cl.CustomComparison(
    output_column_name="address",
    comparison_levels=[
        cll.NullLevel("address_normalized"),
        cll.ExactMatchLevel("address_normalized"),
        # Postcode match
        cll.ExactMatchLevel("postcode"),
        # Fuzzy address match
        cll.LevenshteinLevel("address_normalized", 2),  # 2 char edits
        cll.JaroWinklerLevel("address_normalized", 0.90),
        cll.JaccardLevel("address_normalized", 0.8),  # Token-based
        cll.ElseLevel()
    ]
)
```

### 5.5 Dealing with Very Common vs Very Rare Names

```python
# Automatically handled via term frequency adjustments
name_comparison = cl.NameComparison(
    "name",
    term_frequency_adjustments=True,
    tf_minimum_u_value=0.001  # Prevent extreme weights for rare misspellings
)

# Manual inspection of name frequency
name_counts = df['name'].value_counts()

print("Most common names:")
print(name_counts.head(20))

print("\nRarest names (potential data quality issues):")
print(name_counts.tail(20))

# For very common generic names, require additional evidence
# E.g., "John Smith" - require DOB or address match
common_names = set(name_counts[name_counts > 100].index)

def is_common_name(name):
    return name in common_names

df['is_common_name'] = df['name'].apply(is_common_name)

# In comparisons, use AND logic for common names
name_comparison_enhanced = cl.CustomComparison(
    output_column_name="name_enhanced",
    comparison_levels=[
        cll.NullLevel("name"),
        # Exact match on name
        cll.ExactMatchLevel("name", term_frequency_adjustments=True),
        # For common names, require supporting evidence
        and_(
            cll.JaroWinklerLevel("name", 0.95),
            cll.ExactMatchLevel("dob")  # Must have DOB match too
        ),
        cll.JaroWinklerLevel("name", 0.95, term_frequency_adjustments=True),
        cll.ElseLevel()
    ]
)
```

---

## 6. Performance Optimization

### 6.1 DuckDB vs Spark Backend Selection

**Decision Matrix**:

| Factor | DuckDB | Spark |
|--------|--------|-------|
| **Dataset size** | < 50M records | 50M - 1B+ records |
| **Infrastructure** | Single machine | Distributed cluster |
| **Speed** | Faster (60% faster training, 8.7x faster prediction) | Slower but scalable |
| **Memory** | May OOM on complex linkages | Better memory distribution |
| **Ease of use** | Simple setup | Requires cluster management |
| **Cost** | Low (single machine) | Higher (cluster costs) |

**Code Example**:

```python
# DuckDB (recommended for most cases)
from splink.duckdb.linker import DuckDBLinker

linker = DuckDBLinker(
    df,
    settings,
    connection=None,  # Creates in-memory database
    # OR
    # connection=":temp:",  # On-disk temp database (reduces memory)
)

# Spark (for very large datasets)
from splink.spark.linker import SparkLinker
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Splink") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

df_spark = spark.createDataFrame(df)

linker = SparkLinker(
    df_spark,
    settings,
    spark=spark
)
```

**2025 Update**: DuckDB is recommended for most users and is capable of linking large datasets, especially with high-spec machines. Spark is best for 100M+ records or when you already have a Spark cluster.

### 6.2 Memory Management for Large Datasets

```python
# Strategy 1: Use on-disk DuckDB database
linker = DuckDBLinker(
    df,
    settings,
    connection=":temp:"  # On-disk temporary database
)

# Strategy 2: Tighten blocking rules
# Fewer comparisons = less memory
blocking_rules = [
    # Before (loose): millions of comparisons
    # "substr(l.name, 1, 5) = substr(r.name, 1, 5)",

    # After (tight): fewer comparisons
    "substr(l.name, 1, 15) = substr(r.name, 1, 15)",
]

# Strategy 3: Process in batches (for Spark)
# Use salting to distribute work
blocking_rules = [
    {
        "blocking_rule": "l.surname = r.surname",
        "salting_partitions": 10
    }
]

# Strategy 4: Reduce parallelism in DuckDB
# Trade speed for memory
linker = DuckDBLinker(
    df,
    settings,
    connection=duckdb.connect(config={'threads': 4})  # Limit threads
)

# Strategy 5: Sample for training, full dataset for prediction
df_sample = df.sample(frac=0.5, random_state=42)
linker_training = DuckDBLinker(df_sample, settings)
linker_training.estimate_parameters_using_expectation_maximisation(...)

# Transfer settings to full dataset
linker_full = DuckDBLinker(df, settings)
# Copy trained parameters (m/u values) from linker_training
linker_full._settings_obj = linker_training._settings_obj
```

**Real-World Case** (2025): A user reported DuckDB OOM on 500k records with 1200 GB RAM, while Spark handled 11M records on smaller instances. This suggests memory usage depends heavily on blocking rules and comparison complexity.

### 6.3 Incremental Matching Strategies

**Use Case**: New records arrive daily; don't want to re-run full deduplication.

```python
# Approach 1: Link new records to existing clusters

# Step 1: Dedupe existing data
existing_df = ...  # Historical data
linker_existing = DuckDBLinker(existing_df, settings)
existing_clusters = linker_existing.predict()

# Step 2: Link new data to existing (link_only)
new_df = ...  # New daily records

# Add source indicators
existing_df['source'] = 'existing'
new_df['source'] = 'new'

# Combine
combined_df = pd.concat([existing_df, new_df])

# Use link_only mode
settings_link_only = {
    **settings,
    "link_type": "link_only",  # Only link between datasets, not within
}

linker_incremental = DuckDBLinker(
    [existing_df, new_df],
    settings_link_only
)

new_links = linker_incremental.predict()

# Filter to only new-to-existing links
new_to_existing = new_links[
    (new_links['source_l'] != new_links['source_r'])
]

# Approach 2: Append and dedupe only new segment
# This is a heuristic - assumes new records mostly don't match each other

# Dedupe new records
linker_new = DuckDBLinker(new_df, settings)
new_dedupe_clusters = linker_new.predict()

# Link to existing
# (same as Approach 1)
```

**Caution**: Incremental matching can miss matches between new records if you only link to existing data.

### 6.4 Batch Processing Techniques

```python
# Technique 1: Process in chunks (for very large datasets)

def process_in_batches(df, batch_size=1_000_000):
    n_batches = len(df) // batch_size + 1

    all_predictions = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))

        batch_df = df.iloc[start:end]

        # Dedupe within batch
        linker = DuckDBLinker(batch_df, settings)
        batch_predictions = linker.predict()
        all_predictions.append(batch_predictions)

    # Combine and dedupe across batches
    combined = pd.concat(all_predictions)
    # ... further processing

    return combined

# Technique 2: Use Spark with optimized partitioning

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", 200) \
    .config("spark.default.parallelism", 200) \
    .getOrCreate()

df_spark = spark.createDataFrame(df)

# Repartition for optimal performance
df_spark = df_spark.repartition(200, "surname")  # Partition on blocking key

linker = SparkLinker(df_spark, settings, spark=spark)

# Technique 3: Salting for load balancing
settings = {
    "blocking_rules_to_generate_predictions": [
        {
            "blocking_rule": "l.surname = r.surname",
            "salting_partitions": 120  # High salting for skewed data
        },
        {
            "blocking_rule": "l.postcode = r.postcode",
            "salting_partitions": 80
        }
    ],
    ...
}
```

---

## 7. Advanced Features

### 7.1 Link and Dedupe vs Link Only Strategies

**Three Strategies**:

1. **`dedupe_only`**: Find duplicates within a single dataset
2. **`link_only`**: Link records between multiple datasets (no deduplication within)
3. **`link_and_dedupe`**: Both link between AND dedupe within datasets

**Code Examples**:

```python
# Strategy 1: dedupe_only
# Use case: Single dirty database
settings_dedupe = {
    "link_type": "dedupe_only",
    "comparisons": [...],
    "blocking_rules_to_generate_predictions": [...]
}

linker = DuckDBLinker(df, settings_dedupe)
duplicates = linker.predict()

# Strategy 2: link_only
# Use case: Link two pre-deduplicated datasets
settings_link = {
    "link_type": "link_only",
    "comparisons": [...],
    "blocking_rules_to_generate_predictions": [...]
}

linker = DuckDBLinker([df_customers, df_leads], settings_link)
links = linker.predict()

# Only get cross-dataset links
cross_links = links[links['source_dataset_l'] != links['source_dataset_r']]

# Strategy 3: link_and_dedupe
# Use case: Multiple datasets that may each contain duplicates
settings_both = {
    "link_type": "link_and_dedupe",
    "comparisons": [...],
    "blocking_rules_to_generate_predictions": [...]
}

linker = DuckDBLinker([df1, df2, df3], settings_both)
all_matches = linker.predict()

# This is equivalent to:
# df_union = pd.concat([df1, df2, df3])
# linker = DuckDBLinker(df_union, settings_dedupe_only)
```

**When to Use Multiple Models**:

```python
# Scenario: Two datasets with very different characteristics
# E.g., CRM data (clean) vs web form data (messy)

# Approach 1: Single model (simpler but less accurate)
linker = DuckDBLinker([crm_df, web_df], {
    "link_type": "link_and_dedupe",
    ...
})

# Approach 2: Three models (more complex but more accurate)

# Model 1: Dedupe CRM (clean data → strict matching)
crm_settings = {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.001,  # Low prior (clean data)
    ...
}
linker_crm = DuckDBLinker(crm_df, crm_settings)
crm_deduped = linker_crm.predict()

# Model 2: Dedupe Web (messy data → lenient matching)
web_settings = {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.01,  # Higher prior (messy data)
    ...
}
linker_web = DuckDBLinker(web_df, web_settings)
web_deduped = linker_web.predict()

# Model 3: Link CRM to Web
link_settings = {
    "link_type": "link_only",
    ...
}
linker_link = DuckDBLinker([crm_deduped, web_deduped], link_settings)
cross_links = linker_link.predict()
```

**Guidance**: Use single `link_and_dedupe` if datasets are similar; use three models if datasets have different characteristics or data quality.

### 7.2 Truth Space Analysis

Truth space analysis evaluates model accuracy against ground truth labels.

```python
# Create labeled data
labels = pd.DataFrame([
    {"unique_id_l": "A", "unique_id_r": "B", "clerical_match_score": 1},
    {"unique_id_l": "C", "unique_id_r": "D", "clerical_match_score": 0},
    # ... more labels
])

# Generate truth space table
truth_space = linker.truth_space_table_from_labels_table(labels)

# Visualize accuracy across thresholds
linker.visualisations.accuracy_analysis_from_labels_table(
    labels,
    output_type="threshold_selection",
    out_path="truth_space.html"
)
```

**Metrics Available**:
- Precision and Recall (default)
- Specificity
- Negative Predictive Value (NPV)
- Accuracy
- F₁, F₂, F₀.₅ scores
- P₄ score (addresses F-score limitations)
- φ (Matthews correlation coefficient)

**Use Cases**:
- Threshold selection based on business requirements
- A/B testing model versions
- Identifying where model fails (false positive/negative analysis)

### 7.3 Handling Multiple Data Sources with Different Quality

```python
# Scenario: Linking 3 datasets with different quality levels
# - Gold standard: CRM (high quality, unique IDs)
# - Silver: Partner data (medium quality)
# - Bronze: Web scrape (low quality, messy)

# Strategy 1: Prioritize high-quality data in blocking

# Add data source column
crm_df['data_quality'] = 'gold'
partner_df['data_quality'] = 'silver'
web_df['data_quality'] = 'bronze'

combined_df = pd.concat([crm_df, partner_df, web_df])

# Use stricter blocking for low-quality data
settings = {
    "link_type": "link_and_dedupe",
    "blocking_rules_to_generate_predictions": [
        # Always compare high-quality data broadly
        "l.email = r.email",

        # For bronze data, require stronger evidence
        """
        (l.data_quality = 'bronze' OR r.data_quality = 'bronze')
        AND l.email = r.email
        AND l.surname = r.surname
        """,
    ],
    "comparisons": [...]
}

# Strategy 2: Separate models per quality tier

# Dedupe gold standard
linker_gold = DuckDBLinker(crm_df, {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.0001,  # Very low (high quality)
    ...
})

# Dedupe silver
linker_silver = DuckDBLinker(partner_df, {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.001,
    ...
})

# Dedupe bronze
linker_bronze = DuckDBLinker(web_df, {
    "link_type": "dedupe_only",
    "probability_two_random_records_match": 0.01,  # Higher (low quality)
    ...
})

# Link deduped datasets
# Gold to Silver
linker_gold_silver = DuckDBLinker([gold_deduped, silver_deduped], {
    "link_type": "link_only",
    ...
})

# Silver to Bronze
linker_silver_bronze = DuckDBLinker([silver_deduped, bronze_deduped], {
    "link_type": "link_only",
    # Use stricter thresholds when linking to bronze
    ...
})

# Strategy 3: Weight by data quality in post-processing

predictions['quality_score'] = predictions.apply(
    lambda row: min(
        get_quality_score(row['data_quality_l']),
        get_quality_score(row['data_quality_r'])
    ),
    axis=1
)

# Adjust threshold based on quality
def get_threshold_for_quality(quality_score):
    if quality_score >= 0.9:  # Both gold
        return 0.80
    elif quality_score >= 0.7:  # Gold-silver or silver-silver
        return 0.85
    else:  # Involves bronze
        return 0.92  # Much stricter

predictions['adjusted_match'] = predictions.apply(
    lambda row: row['match_probability'] >= get_threshold_for_quality(row['quality_score']),
    axis=1
)
```

### 7.4 Temporal Matching (Same Entity at Different Time Periods)

**Use Case**: Matching the same entity across snapshots (e.g., customer data from 2020 vs 2025).

```python
# Add temporal context
df['snapshot_year'] = df['record_date'].dt.year

# Feature engineering: Allow for expected changes
def extract_stable_features(row):
    """Features that shouldn't change over time"""
    return {
        'date_of_birth': row['dob'],
        'place_of_birth': row['birthplace'],
        'national_id': row['national_id'],
    }

def extract_changeable_features(row):
    """Features that may change"""
    return {
        'address': row['address'],
        'phone': row['phone'],
        'email': row['email'],
        'employer': row['employer'],
    }

# Temporal comparison strategy
comparisons = [
    # Stable features: Should match exactly
    cl.ExactMatch("date_of_birth"),
    cl.ExactMatch("national_id"),

    # Name: May change (marriage, etc.)
    cl.CustomComparison(
        output_column_name="name",
        comparison_levels=[
            cll.NullLevel("name"),
            cll.ExactMatchLevel("name"),
            cll.JaroWinklerLevel("name", 0.95),
            # Allow partial name match if surnames match
            cll.ComparisonLevel({
                "sql_condition": "l.surname = r.surname",
                "label_for_charts": "Surname match"
            }),
            cll.ElseLevel()
        ]
    ),

    # Address: May change completely
    cl.CustomComparison(
        output_column_name="address",
        comparison_levels=[
            cll.NullLevel("address"),
            cll.ExactMatchLevel("address"),
            # Allow no address match if other evidence is strong
            cll.ElseLevel()
        ]
    ),

    # Temporal consistency checks
    cl.CustomComparison(
        output_column_name="age_consistency",
        comparison_levels=[
            # Age should increase by time difference
            cll.ComparisonLevel({
                "sql_condition": """
                    abs(
                        (l.age - r.age) - (l.snapshot_year - r.snapshot_year)
                    ) <= 1
                """,
                "label_for_charts": "Age progression consistent"
            }),
            cll.ElseLevel()
        ]
    )
]

# Blocking: Don't block on temporal fields
blocking_rules = [
    "l.national_id = r.national_id",
    "l.date_of_birth = r.date_of_birth AND l.surname = r.surname",
    # Note: Not blocking on snapshot_year - we WANT to match across time
]

# Post-processing: Verify temporal logic
def validate_temporal_match(row):
    """Ensure match makes sense temporally"""
    # Same snapshot → dedupe within snapshot
    if row['snapshot_year_l'] == row['snapshot_year_r']:
        return True

    # Different snapshots → verify consistency
    year_diff = abs(row['snapshot_year_l'] - row['snapshot_year_r'])
    age_diff = abs(row['age_l'] - row['age_r'])

    # Age should increase by roughly the year difference
    if abs(age_diff - year_diff) > 2:
        return False  # Inconsistent

    # Don't match if person would be dead
    max_age = max(row['age_l'], row['age_r'])
    if max_age > 120:
        return False

    return True

predictions_df['temporal_valid'] = predictions_df.apply(validate_temporal_match, axis=1)
predictions_df = predictions_df[predictions_df['temporal_valid']]
```

---

## Real-World Examples from 2025

### Harvard Medical School (2025)
Researchers from Harvard Medical School, Vanderbilt University Medical Center and Mass General Brigham used Splink for probabilistic linkage between **8.1 million internet media death records and EHR data**, showing that online obituaries and memorial sites can improve mortality ascertainment by 18-24% over EHRs alone.

### Princeton University (2025)
Researchers from Princeton University, the University of Minnesota, and the Climate and Community Institute used Splink to **link Enterprise-backed multifamily properties to eviction filings and rent listings**, examining how federal mortgage financing relates to rent levels and eviction rates across the US rental market.

### Australian Bureau of Statistics (2025)
The Australian Bureau of Statistics used Splink to build the **2024 National Linkage Spine** and will use Splink for the **2025 Person Linkage Spine build**.

---

## Summary: Production Checklist

Before deploying your Splink model to production:

### Data Preparation
- [ ] Feature engineering complete (phonetic, geographic, temporal features)
- [ ] Address normalization applied
- [ ] Missing data handled appropriately
- [ ] Derived features for hierarchies (parent companies, universities, etc.)

### Model Configuration
- [ ] Blocking rules optimized (not too tight, not too loose)
- [ ] Salting configured for large/skewed datasets
- [ ] Comparison levels appropriate for data quality
- [ ] Term frequency adjustments enabled for name fields
- [ ] Custom comparisons for domain-specific patterns

### Training
- [ ] U probabilities estimated via random sampling
- [ ] Lambda (probability two random records match) estimated
- [ ] EM training converged successfully
- [ ] Multiple training sessions compared (parameter consistency)
- [ ] M/U parameters make intuitive sense

### Evaluation
- [ ] Labeled test set created (50+ pairs)
- [ ] Precision > 95% (or acceptable for use case)
- [ ] Recall > 80% (or acceptable for use case)
- [ ] Waterfall charts reviewed for sample predictions
- [ ] Cluster studio reviewed (no suspicious clusters)
- [ ] ROC/precision-recall curves generated
- [ ] Threshold selected based on business requirements

### Performance
- [ ] Backend selected (DuckDB vs Spark)
- [ ] Memory usage acceptable
- [ ] Runtime acceptable for production schedule
- [ ] Incremental matching strategy defined (if needed)

### False Positive Reduction
- [ ] False positive rate < 5% (on test set)
- [ ] Common false positive patterns identified and addressed
- [ ] Low-density clusters filtered or reviewed
- [ ] Domain-specific rules applied (e.g., university matching for hospitals)

### Monitoring & Maintenance
- [ ] Monitoring plan for ongoing quality
- [ ] Process for handling edge cases
- [ ] Retraining schedule defined
- [ ] Documentation complete

---

## Key Sources

- [Splink Official Documentation](https://moj-analytical-services.github.io/splink/)
- [Phonetic Algorithms Guide](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/phonetic.html)
- [String Comparators](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparators.html)
- [Salting Blocking Rules](https://moj-analytical-services.github.io/splink/topic_guides/performance/salting.html)
- [Training Rationale](https://moj-analytical-services.github.io/splink/topic_guides/training/training_rationale.html)
- [Cluster Studio](https://moj-analytical-services.github.io/splink/charts/cluster_studio_dashboard.html)
- [Term Frequency Adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)
- [Link Types (Link vs Dedupe)](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/link_type.html)
- [DuckDB vs Spark Performance](https://github.com/moj-analytical-services/splink/discussions/2209)
- [Robin Linacre's Blog - EM Intuition](https://www.robinlinacre.com/em_intuition/)
- [Feature Engineering Guide](https://moj-analytical-services.github.io/splink/topic_guides/data_preparation/feature_engineering.html)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-18
**Splink Version**: 3.9.11+
