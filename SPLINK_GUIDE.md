# Splink Organization Matching Guide

## Overview

This guide explains the entity resolution strategy for matching clinical trial organizations using Splink, with a focus on **reducing false positives**.

## Files in This Project

- `splink_organization_matching.ipynb` - Main notebook with full pipeline
- `SPLINK_GUIDE.md` - This guide
- `prod_test/data/test_data.csv` - Input data

## Quick Start

```bash
# Install dependencies
pip install splink pandas duckdb altair

# Run the notebook
jupyter notebook splink_organization_matching.ipynb
```

## Understanding the Data

Your dataset contains Chinese clinical trial organization contacts with:

- **name**: Full organization name
- **metadata**: JSON-like string with:
  - `name_normalized`: Lowercased, cleaned name
  - `name_prefix_5/10`: First 5/10 characters
  - `contact_type`: Scientific/Public
- **name_count**: Frequency of organization appearing in data

### Key Challenge

Many organizations have very similar names:
- "The First Affiliated Hospital of **Zhengzhou** University"
- "The First Affiliated Hospital of **Chongqing** University"

These should NOT match despite high name similarity. This is the main source of false positives.

## Testing Strategy

### 1. Manual Labeled Pairs

Create a ground truth dataset of known matches and non-matches:

```python
labeled_pairs = [
    {
        'name_l': 'Zhongshan Hospital, Fudan University',
        'name_r': 'Zhongshan Hospital Affiliated to Fudan University',
        'clerical_match_score': 1  # TRUE MATCH
    },
    {
        'name_l': 'The First Affiliated Hospital of Zhengzhou University',
        'name_r': 'The First Affiliated Hospital of Chongqing University',
        'clerical_match_score': 0  # FALSE MATCH
    }
]
```

**Target**: Create 50-100 labeled pairs focusing on edge cases.

### 2. Cluster Quality Metrics

Monitor these metrics for each cluster:

- **Density**: Ratio of actual edges to possible edges
  - Good: > 0.7 (most pairs in cluster match)
  - Suspicious: < 0.5 (sparse connections suggest false positives)

- **Cluster Centralization**: Distribution of connections
  - Good: < 0.5 (well-distributed matches)
  - Suspicious: > 0.7 (star pattern, one record connected to many)

### 3. Threshold Analysis

Test multiple match probability thresholds:

| Threshold | Expected Outcome |
|-----------|------------------|
| 0.75      | High recall, more false positives |
| 0.85      | Balanced (recommended starting point) |
| 0.90      | High precision, fewer false positives |
| 0.95      | Very conservative, may miss some matches |

### 4. Sampling Strategy

Randomly sample 20-30 clusters and manually validate:
1. All members should represent the same real-world organization
2. Check large clusters (5+ members) - more prone to false positives
3. Review clusters with low density scores

## Measuring Improvement

### Baseline Metrics (Before Tuning)

Run the notebook and record:
```
Total matches at threshold 0.85: ___
Number of clusters: ___
Average cluster size: ___
Large clusters (5+ members): ___
Low density clusters (<0.5): ___
```

### After Each Change

Compare:
1. **Precision**: % of predicted matches that are correct
   - Manually validate sample of 20 matches
   - Calculate: True Positives / (True Positives + False Positives)

2. **Recall**: % of true matches found
   - Check against known duplicates
   - Calculate: True Positives / (True Positives + False Negatives)

3. **F1 Score**: Harmonic mean of precision and recall
   - Calculate: 2 * (Precision * Recall) / (Precision + Recall)

### Key Performance Indicators

| Metric | Target | Priority |
|--------|--------|----------|
| False Positive Rate | < 5% | HIGH |
| Recall | > 80% | MEDIUM |
| Cluster Density (avg) | > 0.7 | HIGH |
| Large Clusters | Minimize | HIGH |

## Reducing False Positives: Techniques

### 1. Increase Match Threshold ✅ Implemented

**Current**: 0.85
**Try**: 0.90 or 0.95

```python
MATCH_THRESHOLD = 0.90  # Change this in the notebook
```

**Effect**: Reduces false positives but may miss some true matches.

### 2. Stricter Comparison Levels ✅ Implemented

**Current approach**:
- Multiple Jaro-Winkler thresholds (0.95, 0.92, 0.88, 0.85)
- Term frequency adjustments

**To make stricter**:
```python
# Remove the 0.85 and 0.88 levels, only keep high-confidence
comparison_levels=[
    cll.ExactMatchLevel("name_clean", term_frequency_adjustments=True),
    cll.JaroWinklerLevel("name_clean", 0.95, term_frequency_adjustments=True),
    cll.JaroWinklerLevel("name_clean", 0.92, term_frequency_adjustments=True),
    cll.ElseLevel()
]
```

### 3. Require Supporting Evidence ⚠️ Partially Implemented

**Current**: University name and hospital order are compared but optional
**Improvement**: Make university name a required match

```python
# In blocking rules, add:
"l.university_name = r.university_name"

# This ensures pairs MUST have same university to match
```

### 4. Extract More Distinguishing Features 🔄 Recommended

Add these features to improve discrimination:

```python
# Extract city/location
def extract_city(name):
    # Parse from name or metadata
    # e.g., "Beijing Tiantan Hospital" -> "Beijing"
    pass

# Extract hospital type
def extract_type(name):
    # "Children's Medical Center" -> "Children"
    # "Cancer Hospital" -> "Cancer"
    pass
```

### 5. Tighter Blocking Rules ✅ Implemented

**Current**: First 15 characters must match exactly

**To make tighter**:
```python
# Increase from 15 to 20 characters
"substr(l.name_clean, 1, 20) = substr(r.name_clean, 1, 20)"

# Add multiple required conditions
"substr(l.name_clean, 1, 15) = substr(r.name_clean, 1, 15) AND l.university_name = r.university_name"
```

**Trade-off**: Tighter blocking may miss matches with typos or variations.

### 6. Manual Rules for Problem Patterns 🆕 Add This

Identify specific patterns causing false positives:

```python
# Example: Don't match if universities differ
# This can be a post-processing filter
def filter_false_positives(predictions_df):
    # Remove matches where university names are different
    predictions_df = predictions_df[
        (predictions_df['university_name_l'] == predictions_df['university_name_r']) |
        (predictions_df['university_name_l'].isna()) |
        (predictions_df['university_name_r'].isna())
    ]
    return predictions_df
```

### 7. Use Cluster Metrics for Filtering 🆕 Add This

```python
# Filter out suspicious clusters
def filter_low_quality_clusters(clusters_df, metrics_df, min_density=0.6):
    \"\"\"Remove clusters with low density\"\"\"
    good_clusters = metrics_df[
        (metrics_df['density'] >= min_density) |
        (metrics_df['n_nodes'] == 2)  # Pairs are ok
    ]['cluster_id']

    return clusters_df[clusters_df['cluster_id'].isin(good_clusters)]
```

## Common False Positive Patterns

### Pattern 1: Same Hospital Type, Different Location

**Example**:
- "The First Affiliated Hospital of **Zhengzhou** University"
- "The First Affiliated Hospital of **Chongqing** University"

**Why it happens**: Very high name similarity (Jaro-Winkler ~0.90)

**Solution**: Require university name match OR extract location

### Pattern 2: Generic Hospital Names

**Example**:
- "Affiliated Hospital of **Xuzhou** Medical University"
- "Affiliated Hospital of **Guangdong** Medical University"

**Why it happens**: Common prefix "Affiliated Hospital of"

**Solution**:
- Term frequency adjustments (✅ implemented)
- Require exact match on university name
- Increase minimum threshold

### Pattern 3: Similar Universities

**Example**:
- "Sun Yat-sen Memorial Hospital, **Sun Yat-sen University**"
- "The First Affiliated Hospital of **Sun Yat-Sen University**" (note: Sen vs sen)

**Why it happens**: Same university, different hospital

**Solution**:
- Extract and compare hospital name component
- Not actually a false positive if same university!

## Workflow: Iterative Improvement

```
1. Run baseline model (threshold 0.85)
   ↓
2. Manually validate 20 random matches
   ↓
3. Calculate precision: TP / (TP + FP)
   ↓
4. If precision < 95%:
   → Identify false positive patterns
   → Apply one technique from above
   → Re-run and measure
   ↓
5. Check recall on known duplicates
   ↓
6. If recall < 80%:
   → Lower threshold OR
   → Relax blocking rules
   ↓
7. Repeat until satisfied
```

## Advanced: Custom Comparison Function

For maximum control, create a custom comparison that requires both name similarity AND university match:

```python
from splink.comparison_level_composition import and_

custom_level = and_(
    cll.JaroWinklerLevel("name_clean", 0.90),
    cll.ExactMatchLevel("university_name")
)
```

## Validation Checklist

Before deploying your model:

- [ ] Manually validated at least 50 predictions
- [ ] False positive rate < 5%
- [ ] Recall rate > 80% on known duplicates
- [ ] Reviewed all clusters with 5+ members
- [ ] Reviewed all clusters with density < 0.6
- [ ] Tested on held-out sample (if available)
- [ ] Documented threshold and parameter choices
- [ ] Created monitoring plan for production

## Troubleshooting

### "Too many false positives"

1. Increase `MATCH_THRESHOLD` to 0.90
2. Remove low Jaro-Winkler levels (0.85, 0.88)
3. Add required university name match to blocking
4. Filter clusters with density < 0.6

### "Missing obvious matches"

1. Lower `MATCH_THRESHOLD` to 0.80
2. Add more blocking rules (shorter prefixes)
3. Check if names are properly normalized
4. Add Levenshtein comparison for typos

### "Model training is slow"

1. Use stricter blocking for training
2. Reduce data size for parameter estimation
3. Use fewer comparison levels initially
4. Consider using DuckDB backend (already implemented)

### "Clusters are too large"

1. Increase match threshold
2. This often indicates false positive chains
3. Review cluster metrics - look for low density
4. Add post-processing to split clusters

## References

- [Splink Documentation](https://moj-analytical-services.github.io/splink/)
- [Splink GitHub](https://github.com/moj-analytical-services/splink)
- [Blocking Rules Guide](https://moj-analytical-services.github.io/splink/topic_guides/blocking/blocking_rules.html)
- [Evaluation Metrics](https://moj-analytical-services.github.io/splink/topic_guides/evaluation/edge_metrics.html)

## Support

For questions or issues:
1. Check the notebook comments and markdown cells
2. Review the Splink documentation
3. Examine the waterfall charts for specific predictions
4. Use the cluster studio for interactive exploration
