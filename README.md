# Splink Organization Entity Resolution

A comprehensive entity resolution solution for matching clinical trial organizations using Splink, with a strong focus on reducing false positives.

## Overview

This project provides a complete pipeline for:
- Deduplicating organization records from clinical trial data
- Identifying which records represent the same real-world entity
- Generating new records vs resolving to existing ones
- Measuring and improving matching accuracy

## Problem Context

The data contains Chinese clinical trial organization contacts with patterns like:
- "The First Affiliated Hospital of **Zhengzhou** University"
- "The First Affiliated Hospital of **Chongqing** University"

These names are very similar but represent **different organizations**. The challenge is distinguishing between:
- **True duplicates**: "Zhongshan Hospital, Fudan University" ↔ "Zhongshan Hospital Affiliated to Fudan University" ✓
- **False positives**: "First Hospital of University A" ↔ "First Hospital of University B" ✗

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook splink_organization_matching.ipynb
```

### 3. Follow the Workflow

The notebook guides you through:
1. Data loading and preparation
2. Feature engineering
3. Model configuration
4. Training with Expectation Maximization
5. Generating predictions
6. Cluster analysis
7. Evaluation and metrics
8. Threshold tuning

## Files

| File | Purpose |
|------|---------|
| `splink_organization_matching.ipynb` | **Main notebook** - Complete entity resolution pipeline |
| `SPLINK_GUIDE.md` | **Comprehensive guide** - Best practices, troubleshooting, patterns |
| `evaluation_tools.py` | **Helper utilities** - Evaluation, metrics, testing |
| `model_config.py` | **Configuration** - Tune parameters without editing notebook |
| `requirements.txt` | Python dependencies |
| `prod_test/data/test_data.csv` | Input data |

## Key Features

### 🎯 False Positive Reduction

Multiple strategies implemented:
- **High match thresholds** (default: 0.85)
- **Multi-level fuzzy matching** with Jaro-Winkler
- **Term frequency adjustments** for common vs uncommon names
- **Cluster quality metrics** to identify suspicious matches
- **Configurable blocking rules** for precision control

### 📊 Testing Strategy

1. **Manual labeled pairs** - Ground truth dataset
2. **Cluster quality metrics** - Density and centralization
3. **Threshold analysis** - Compare multiple thresholds
4. **Stratified sampling** - Review high/medium/low confidence matches

### 📈 Evaluation Metrics

- **Precision**: % of predicted matches that are correct (minimize false positives)
- **Recall**: % of true matches found (don't miss duplicates)
- **F1 Score**: Harmonic mean of precision and recall
- **Cluster Density**: Quality of clusters (>0.7 is good)
- **Cluster Centralization**: Distribution of connections (<0.5 is good)

### ⚙️ Easy Configuration

Three preset configurations:

```python
# Minimize false positives (recommended if you had FP issues before)
config = apply_preset('conservative')

# Balanced precision and recall (default)
config = apply_preset('balanced')

# Maximize recall (you'll manually filter FPs)
config = apply_preset('aggressive')
```

## Usage Example

```python
from evaluation_tools import MatchEvaluator, ThresholdOptimizer

# After running the notebook...

# Create evaluator
evaluator = MatchEvaluator(predictions_df, clusters_df)

# Export sample for manual review
evaluator.export_review_sample('review_sample.csv', n_samples=30)

# Analyze cluster quality
quality = evaluator.analyze_cluster_quality(metrics_df)
print(f"Suspicious clusters: {quality['suspicious_clusters']}")

# Optimize threshold (if you have labels)
optimizer = ThresholdOptimizer(predictions_df, labels_df)
best_threshold, metrics, results = optimizer.find_optimal_threshold(optimize_for='f1')
print(f"Best threshold: {best_threshold} (F1: {metrics['f1']:.3f})")
```

## Measuring Improvement

### Baseline (First Run)

1. Run notebook with default settings
2. Record metrics:
   ```
   Total matches: ___
   Clusters: ___
   Large clusters (5+): ___
   Low density clusters: ___
   ```

3. Manually validate 20-30 random matches
4. Calculate precision: True Matches / Total Sampled

### After Tuning

1. Apply one change (e.g., increase threshold to 0.90)
2. Re-run and compare metrics
3. Re-validate sample
4. Calculate improvement:
   ```
   Precision improved from 85% → 95% ✓
   Recall maintained at 80% ✓
   False positives reduced by 60% ✓
   ```

### Target Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| False Positive Rate | < 5% | **HIGH** |
| Recall | > 80% | MEDIUM |
| Avg Cluster Density | > 0.7 | **HIGH** |
| Suspicious Clusters | Minimize | **HIGH** |

## Common False Positive Patterns

Based on the data, watch for:

### 1. Different Universities, Same Hospital Type
```
"The First Affiliated Hospital of Zhengzhou University"
"The First Affiliated Hospital of Chongqing University"
→ Should NOT match (different organizations)
```

**Solution**: Enable `REQUIRE_UNIVERSITY_MATCH_FINAL` in `model_config.py`

### 2. Generic Names
```
"Affiliated Hospital of Xuzhou Medical University"
"Affiliated Hospital of Guangdong Medical University"
→ Should NOT match
```

**Solution**: Increase `MATCH_THRESHOLD` to 0.90+

### 3. Spelling Variations (True Matches!)
```
"Sun Yat-sen University"
"Sun Yat-Sen University"  (different capitalization)
→ SHOULD match (same organization)
```

**Solution**: Ensure name normalization is working

## Workflow: Iterative Improvement

```
┌─────────────────────────────────────────┐
│ 1. Run baseline (threshold 0.85)       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Sample 30 matches                    │
│    - 10 high confidence (>0.95)         │
│    - 10 medium (0.85-0.95)              │
│    - 10 low (0.75-0.85)                 │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 3. Manually validate                    │
│    Calculate: Precision = TP/(TP+FP)    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 4. If precision < 95%:                  │
│    → Identify FP patterns               │
│    → Increase threshold OR              │
│    → Enable university matching OR      │
│    → Tighten blocking rules             │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 5. Re-run and measure                   │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 6. Repeat until satisfied               │
└─────────────────────────────────────────┘
```

## Advanced: Tuning Parameters

Edit `model_config.py` to adjust:

```python
# More conservative (fewer false positives)
MATCH_THRESHOLD = 0.90  # Increase from 0.85
JARO_WINKLER_LOW = 0.90  # Increase from 0.88
PREFIX_MATCH_LENGTH_LONG = 20  # Increase from 15
REQUIRE_UNIVERSITY_MATCH_FINAL = True  # Enable filter

# More aggressive (catch more matches)
MATCH_THRESHOLD = 0.80  # Decrease
JARO_WINKLER_VERY_LOW = 0.80  # Add lower level
PREFIX_MATCH_LENGTH_LONG = 10  # Decrease for more comparisons
```

## Troubleshooting

### "Too many false positives"

1. ✓ Increase `MATCH_THRESHOLD` to 0.90
2. ✓ Apply conservative preset: `apply_preset('conservative')`
3. ✓ Enable `REQUIRE_UNIVERSITY_MATCH_FINAL = True`
4. ✓ Filter clusters with density < 0.6

### "Missing obvious matches"

1. ✓ Lower `MATCH_THRESHOLD` to 0.80
2. ✓ Check name normalization is working
3. ✓ Add more blocking rules
4. ✓ Review false negatives in evaluation

### "Model is too slow"

1. ✓ Increase `PREFIX_MATCH_LENGTH_LONG` (fewer comparisons)
2. ✓ Use tighter blocking rules
3. ✓ DuckDB backend is already used (fast)

## Research Background

This implementation is based on:
- [Splink Documentation](https://moj-analytical-services.github.io/splink/)
- Best practices for fuzzy name matching
- Term frequency adjustments for entity resolution
- Probabilistic linkage with EM training
- Cluster analysis for match quality

Key insights from research:
1. Jaro-Winkler ≥ 0.88 recommended for high precision
2. Multi-level comparisons better than single threshold
3. Term frequency adjustments critical for common names
4. Exact matching in blocking, fuzzy in comparisons
5. Cluster metrics reveal false positive patterns

## Next Steps

1. ✓ Run the notebook end-to-end
2. ✓ Review suspicious clusters identified
3. ✓ Validate 30 random matches
4. ✓ Calculate baseline precision/recall
5. ✓ Apply one tuning technique
6. ✓ Measure improvement
7. ✓ Iterate until metrics meet targets
8. ✓ Document final configuration

## Support

For issues or questions:
1. Review `SPLINK_GUIDE.md` for detailed troubleshooting
2. Check the notebook markdown cells
3. Examine Splink documentation: https://moj-analytical-services.github.io/splink/
4. Use waterfall charts in notebook to debug specific matches

## License

This project uses Splink, which is licensed under MIT License.

---

**Created for**: Organization entity resolution in clinical trial data
**Focus**: Minimizing false positives while maintaining high recall
**Approach**: Probabilistic linkage with Expectation Maximization
