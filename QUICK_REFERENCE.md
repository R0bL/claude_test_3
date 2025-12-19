# Splink Advanced Techniques - Quick Reference

**Last Updated**: 2025-12-18
**Splink Version**: 3.9.11+

This is a quick reference companion to `ADVANCED_SPLINK_TECHNIQUES.md`. For detailed explanations and examples, see the main guide.

---

## Files in This Repository

| File | Purpose |
|------|---------|
| `ADVANCED_SPLINK_TECHNIQUES.md` | Comprehensive guide with all techniques |
| `advanced_comparison_library.py` | Production-ready comparison functions |
| `advanced_blocking_optimization.py` | Blocking rules and performance optimization |
| `corporate_matching.py` | Corporate entity matching extensions |
| `model_config.py` | Configuration presets |
| `evaluation_tools.py` | Evaluation utilities |
| `QUICK_REFERENCE.md` | This file |

---

## Comparison Functions Cheat Sheet

### Phonetic Matching

```python
from advanced_comparison_library import PhoneticComparisons

# 1. Prepare data
df = PhoneticComparisons.prepare_phonetic_columns(df, 'name')

# 2. Create comparison
comparison = PhoneticComparisons.get_phonetic_comparison_dict(
    'name',
    include_dmeta=True,      # Double Metaphone
    include_soundex=False,    # Soundex
    include_metaphone=False   # Metaphone
)

settings["comparisons"].append(comparison)
```

### Geographic Distance

```python
from advanced_comparison_library import GeographicComparisons

comparison = GeographicComparisons.get_distance_comparison_dict(
    lat_col="latitude",
    lon_col="longitude",
    km_thresholds=[1, 10, 50, 100]  # kilometers
)
```

### Fuzzy Dates

```python
from advanced_comparison_library import DateComparisons

comparison = DateComparisons.get_date_comparison_dict(
    date_col="date_of_birth",
    thresholds_days=[0, 1, 30, 365],  # exact, 1 day, 1 month, 1 year
    allow_year_only=True
)
```

### Array/Alias Matching

```python
from advanced_comparison_library import ArrayComparisons

# Prepare data
df = ArrayComparisons.prepare_alias_columns(
    df,
    alias_columns=['official_name', 'alias1', 'alias2'],
    output_col='all_names'
)

# Exact array matching
comparison = ArrayComparisons.get_array_exact_comparison_dict(
    'all_names',
    min_intersections=[3, 1]
)

# Fuzzy array matching
comparison = ArrayComparisons.get_array_fuzzy_comparison_dict(
    'all_names',
    similarity_threshold=0.9
)
```

### Jaccard (Token-Based)

```python
from advanced_comparison_library import TokenComparisons

# Best for addresses and multi-word strings
comparison = TokenComparisons.get_jaccard_comparison_dict(
    'address',
    thresholds=[0.9, 0.7, 0.5]
)
```

### Hospital/University (Prevents False Positives)

```python
from advanced_comparison_library import CompositeComparisons

comparison = CompositeComparisons.get_hospital_university_comparison_dict(
    name_col='name',
    university_col='university_name',
    jw_thresholds=[0.95, 0.90]
)
```

---

## Blocking Rules Cheat Sheet

### Basic Rules

```python
from advanced_blocking_optimization import BlockingRuleGenerator

gen = BlockingRuleGenerator()

# Exact match
gen.exact_match_rule("email")
# → "l.email = r.email"

# Multi-column exact match
gen.multi_column_exact_match(["surname", "dob"])
# → "l.surname = r.surname AND l.dob = r.dob"

# Prefix match
gen.prefix_match_rule("surname", 5)
# → "substr(l.surname, 1, 5) = substr(r.surname, 1, 5)"

# Prefix + exact
gen.prefix_and_exact_rule("surname", 5, "city")
# → "substr(l.surname, 1, 5) = substr(r.surname, 1, 5) AND l.city = r.city"
```

### Pre-Built Configurations

```python
from advanced_blocking_optimization import BlockingConfigs

# Person matching
blocking_rules = BlockingConfigs.person_matching_rules()

# Organization matching
blocking_rules = BlockingConfigs.organization_matching_rules()

# Hospital/university matching
blocking_rules = BlockingConfigs.hospital_university_matching_rules()
```

### Salting for Performance

```python
from advanced_blocking_optimization import SaltingStrategy

salting = SaltingStrategy()

# Estimate partitions
num_partitions = salting.estimate_salting_partitions(
    df,
    blocking_col='surname',
    target_comparisons_per_partition=1_000_000
)

# Create salted rule
salted_rule = salting.create_salted_rule(
    blocking_rule="l.surname = r.surname",
    num_partitions=10
)
# → {"blocking_rule": "l.surname = r.surname", "salting_partitions": 10}

# Adaptive salting (automatically detects skew)
blocking_rules = ["l.name = r.name", "l.city = r.city"]
blocking_cols = ["name", "city"]

salted_rules = salting.get_adaptive_salting_config(
    blocking_rules, df, blocking_cols
)
```

### Blocking Analysis

```python
from advanced_blocking_optimization import BlockingOptimizer

optimizer = BlockingOptimizer()

# Analyze coverage
analysis = optimizer.analyze_blocking_rule_coverage(
    df,
    blocking_rules=["l.name = r.name", "l.city = r.city"],
    blocking_cols=["name", "city"]
)
print(analysis)
# Shows estimated_comparisons, pct_of_total_pairs, skew_ratio, etc.

# Auto-suggest rules
suggested_rules = optimizer.suggest_blocking_rules(
    df,
    candidate_columns=['name', 'dob', 'email', 'city'],
    max_rules=5
)
```

---

## Model Training Cheat Sheet

### Standard EM Training (Recommended)

```python
from splink.duckdb.linker import DuckDBLinker

linker = DuckDBLinker(df, settings)

# Step 1: Estimate u (random pairs)
linker.estimate_u_using_random_sampling(max_pairs=1e7)

# Step 2: Estimate lambda (prior)
linker.estimate_probability_two_random_records_match(
    deterministic_matching_rules=[
        "l.email = r.email",
        "l.phone = r.phone AND l.surname = r.surname"
    ],
    recall=0.7  # Conservative
)

# Step 3: EM training for m (multiple sessions)
session_1 = linker.estimate_m_from_blocking_rule(
    "l.surname = r.surname",
    fix_u_probabilities=True,
    fix_probability_two_random_records_match=True
)

session_2 = linker.estimate_m_from_blocking_rule(
    "l.postcode = r.postcode",
    fix_u_probabilities=True,
    fix_probability_two_random_records_match=True
)

# Step 4: Compare estimates
linker.visualisations.parameter_estimate_comparisons_chart(
    [session_1, session_2],
    out_path="param_comparison.html"
)
```

### Training vs Prediction Rules

```python
from advanced_blocking_optimization import PerformanceTuner

tuner = PerformanceTuner()

base_rules = [
    "substr(l.surname, 1, 10) = substr(r.surname, 1, 10)",
    "l.city = r.city"
]

training_rules, prediction_rules = tuner.get_training_vs_prediction_blocking_rules(
    base_rules,
    training_multiplier=0.5  # Make training rules tighter
)

# Use training_rules for estimate_m_from_blocking_rule
# Use prediction_rules for predict()
```

---

## Evaluation Cheat Sheet

### Visualizations

```python
# Waterfall chart (individual predictions)
linker.visualisations.waterfall_chart(
    records=[record_id_1, record_id_2],
    out_path="waterfall.html"
)

# M/U parameters
linker.visualisations.m_u_parameters_chart(
    out_path="m_u.html"
)

# Match weights distribution
linker.visualisations.match_weights_chart(
    predictions_df,
    out_path="match_weights.html"
)
```

### Cluster Studio

```python
from splink import cluster_studio

clusters = linker.cluster_pairwise_predictions_at_threshold(
    predictions_df,
    threshold_match_probability=0.85
)

cluster_studio.create_cluster_studio_dashboard(
    predictions_df=predictions_df,
    clusters_df=clusters,
    out_path="cluster_studio.html",
    sample_size=100
)
```

### Precision/Recall Analysis

```python
# Create labels
labels = pd.DataFrame([
    {"unique_id_l": "A", "unique_id_r": "B", "clerical_match_score": 1},
    {"unique_id_l": "C", "unique_id_r": "D", "clerical_match_score": 0},
])

# Threshold selection tool
linker.visualisations.threshold_selection_tool_from_labels_table(
    labels,
    out_path="threshold_tool.html"
)

# ROC curve
linker.visualisations.accuracy_analysis_from_labels_table(
    labels,
    output_type="roc",
    out_path="roc.html"
)

# Precision-Recall curve
linker.visualisations.accuracy_analysis_from_labels_table(
    labels,
    output_type="threshold_selection",
    out_path="pr_curve.html"
)
```

---

## Performance Optimization Cheat Sheet

### DuckDB vs Spark

| Factor | DuckDB | Spark |
|--------|--------|-------|
| Best for | < 50M records | 50M+ records |
| Speed | 60% faster training, 8.7x faster prediction | Slower but scalable |
| Setup | Simple | Complex (cluster) |
| Memory | May OOM on complex jobs | Better distribution |

```python
# DuckDB (default)
from splink.duckdb.linker import DuckDBLinker
linker = DuckDBLinker(df, settings)

# DuckDB (on-disk for memory)
linker = DuckDBLinker(df, settings, connection=":temp:")

# Spark
from splink.spark.linker import SparkLinker
linker = SparkLinker(df_spark, settings, spark=spark)
```

### Memory Optimization

```python
# 1. Tighter blocking rules
blocking_rules = [
    "substr(l.name, 1, 15) = substr(r.name, 1, 15)"  # Was 5
]

# 2. On-disk database
linker = DuckDBLinker(df, settings, connection=":temp:")

# 3. Limit parallelism
import duckdb
conn = duckdb.connect(config={'threads': 4})
linker = DuckDBLinker(df, settings, connection=conn)

# 4. Salting (Spark only)
blocking_rules = [
    {
        "blocking_rule": "l.surname = r.surname",
        "salting_partitions": 10
    }
]
```

---

## Real-World Patterns Cheat Sheet

### Corporate Entities

```python
from corporate_matching import (
    prepare_corporate_data,
    create_corporate_blocking_rules,
    post_process_corporate_matches
)

# 1. Feature engineering
df = prepare_corporate_data(df, name_column='name')
# Adds: country_code_from_name, parent_company, core_company_name, etc.

# 2. Blocking rules
blocking_rules = create_corporate_blocking_rules()

# 3. Post-processing
filtered_predictions = post_process_corporate_matches(
    predictions_df,
    df,
    match_level="entity"  # or "parent" to include same-parent matches
)
```

### Hospitals/Universities

```python
# CRITICAL: Always require university match to prevent FPs

# Feature extraction
def extract_university(name):
    patterns = [
        r'of ([^,]+University[^,]*)',
        r'Affiliated to ([^,]+University[^,]*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return match.group(1).strip()
    return None

df['university_name'] = df['name'].apply(extract_university)

# Use hospital/university comparison (requires both name + university match)
from advanced_comparison_library import CompositeComparisons

comparison = CompositeComparisons.get_hospital_university_comparison_dict(
    name_col='name',
    university_col='university_name'
)
```

### Address Normalization

```python
import re

def normalize_address(addr):
    if pd.isna(addr):
        return None

    addr = addr.lower().strip()
    addr = re.sub(r'[.,]', ' ', addr)

    replacements = {
        r'\bstreet\b': 'st',
        r'\broad\b': 'rd',
        r'\bavenue\b': 'ave',
        r'\bnorth\b': 'n',
        r'\bsouth\b': 's',
        r'\beast\b': 'e',
        r'\bwest\b': 'w',
    }

    for pattern, replacement in replacements.items():
        addr = re.sub(pattern, replacement, addr)

    addr = re.sub(r'\s+', ' ', addr)
    return addr.strip()

df['address_normalized'] = df['address'].apply(normalize_address)
```

---

## Common Pitfalls & Solutions

### Problem: False Positives on Generic Names

**Example**: "First Affiliated Hospital of X" matching "First Affiliated Hospital of Y"

**Solution**:
```python
# Use AND logic requiring disambiguating field
comparison = CompositeComparisons.get_hospital_university_comparison_dict(
    name_col='name',
    university_col='university_name'  # MUST match
)
```

### Problem: Very Common Names (John Smith)

**Solution**:
```python
# 1. Use term frequency adjustments (automatic)
name_comparison = cl.NameComparison(
    "name",
    term_frequency_adjustments=True
)

# 2. Require additional evidence
comparison = {
    "output_column_name": "name_enhanced",
    "comparison_levels": [
        cll.NullLevel("name"),
        cll.ExactMatchLevel("name", term_frequency_adjustments=True),
        # For fuzzy match, require DOB support
        {
            "sql_condition": """
                jaro_winkler_similarity(l.name, r.name) >= 0.95
                AND l.dob = r.dob
            """,
            "label_for_charts": "Fuzzy name + DOB match"
        },
        cll.ElseLevel()
    ]
}
```

### Problem: Out of Memory

**Solutions**:
1. Tighten blocking rules (reduce comparisons)
2. Use on-disk DuckDB: `connection=":temp:"`
3. Switch to Spark for very large datasets
4. Use salting (Spark)
5. Process in batches

### Problem: Training Not Converging

**Solutions**:
```python
# 1. Fix u and lambda (don't let EM estimate these)
linker.estimate_u_using_random_sampling(max_pairs=1e7)
linker.estimate_probability_two_random_records_match([...], recall=0.7)

# 2. Use tight blocking for training
linker.estimate_m_from_blocking_rule(
    "l.surname = r.surname AND l.dob = r.dob",  # Tight
    fix_u_probabilities=True,
    fix_probability_two_random_records_match=True
)

# 3. Compare multiple training sessions
# If estimates are very different, investigate blocking rule bias
```

### Problem: Missing True Matches (Low Recall)

**Solutions**:
1. Add more/broader blocking rules
2. Lower match threshold
3. Use phonetic matching for name variations
4. Check blocking rule coverage:
   ```python
   optimizer.analyze_blocking_rule_coverage(df, blocking_rules, blocking_cols)
   ```

---

## Threshold Selection Guide

| Use Case | Optimize For | Typical Threshold | Acceptable FP Rate |
|----------|-------------|-------------------|-------------------|
| Medical records | Precision | 0.92 - 0.95 | < 1% |
| Financial data | Precision | 0.90 - 0.95 | < 2% |
| Customer deduplication | F₁ score | 0.85 - 0.90 | < 5% |
| Marketing lists | Recall | 0.75 - 0.85 | < 10% |
| Candidate generation | Recall | 0.70 - 0.80 | 10-20% (manual review) |

**How to select**:
```python
# 1. Use threshold selection tool
linker.visualisations.threshold_selection_tool_from_labels_table(
    labels,
    out_path="threshold_tool.html"
)

# 2. Or programmatically
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# For 95% precision
idx = np.argmax(precision >= 0.95)
threshold_for_95_precision = thresholds[idx]
```

---

## Link Type Strategy

### dedupe_only
```python
settings = {
    "link_type": "dedupe_only",
    ...
}
linker = DuckDBLinker(df, settings)
```
**Use**: Single dirty database

### link_only
```python
settings = {
    "link_type": "link_only",
    ...
}
linker = DuckDBLinker([df1, df2], settings)
```
**Use**: Link two pre-deduplicated datasets

### link_and_dedupe
```python
settings = {
    "link_type": "link_and_dedupe",
    ...
}
linker = DuckDBLinker([df1, df2, df3], settings)
```
**Use**: Multiple datasets that may each contain duplicates

### Multi-Model Strategy
**When to use**: Datasets with very different characteristics

```python
# Model 1: Dedupe CRM (clean data)
linker_crm = DuckDBLinker(crm_df, crm_settings)
crm_deduped = linker_crm.predict()

# Model 2: Dedupe Web (messy data)
linker_web = DuckDBLinker(web_df, web_settings)
web_deduped = linker_web.predict()

# Model 3: Link CRM to Web
linker_link = DuckDBLinker([crm_deduped, web_deduped], link_settings)
cross_links = linker_link.predict()
```

---

## Production Deployment Checklist

### Before Production

- [ ] Labeled test set created (50-100 pairs)
- [ ] Precision > 95% (or acceptable for use case)
- [ ] Recall > 80% (or acceptable for use case)
- [ ] Waterfall charts reviewed for sample predictions
- [ ] Cluster studio reviewed (no suspicious clusters)
- [ ] Parameter estimates consistent across training sessions
- [ ] M/U parameters make intuitive sense
- [ ] Blocking rules optimized (coverage analysis done)
- [ ] Performance acceptable (runtime estimation done)
- [ ] False positive patterns identified and addressed
- [ ] Documentation complete

### Monitoring in Production

- [ ] Sample random predictions monthly
- [ ] Track precision/recall on new labels
- [ ] Monitor cluster size distribution
- [ ] Watch for new false positive patterns
- [ ] Re-train quarterly (or when data distribution changes)
- [ ] Log runtime and comparisons count

---

## Key Resources

- **Official Docs**: https://moj-analytical-services.github.io/splink/
- **GitHub**: https://github.com/moj-analytical-services/splink
- **Robin Linacre's Blog**: https://www.robinlinacre.com/
- **Community**: GitHub Discussions

## Common Imports

```python
# Core
from splink.duckdb.linker import DuckDBLinker
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_level_library as cll

# Custom
from advanced_comparison_library import (
    PhoneticComparisons,
    GeographicComparisons,
    DateComparisons,
    ArrayComparisons,
    CompositeComparisons
)

from advanced_blocking_optimization import (
    BlockingRuleGenerator,
    SaltingStrategy,
    BlockingOptimizer,
    BlockingConfigs
)

from corporate_matching import (
    prepare_corporate_data,
    create_corporate_blocking_rules
)
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-18
