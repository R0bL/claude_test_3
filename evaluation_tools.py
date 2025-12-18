"""
Evaluation Tools for Splink Organization Matching

This module provides utilities for:
1. Generating evaluation metrics
2. Analyzing false positives
3. Creating test datasets
4. Comparing model versions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json


class MatchEvaluator:
    """Evaluate quality of entity resolution results"""

    def __init__(self, predictions_df: pd.DataFrame, clusters_df: pd.DataFrame):
        """
        Initialize evaluator

        Args:
            predictions_df: Pairwise predictions from Splink
            clusters_df: Clustered results from Splink
        """
        self.predictions = predictions_df
        self.clusters = clusters_df

    def calculate_precision_from_sample(self, sample_size: int = 20, random_state: int = 42) -> Dict:
        """
        Calculate precision by manually validating a random sample

        Args:
            sample_size: Number of predictions to sample
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with sample and instructions
        """
        sample = self.predictions.nlargest(sample_size, 'match_probability')

        # Create validation template
        validation_template = []
        for idx, row in sample.iterrows():
            validation_template.append({
                'name_l': row['name_l'],
                'name_r': row['name_r'],
                'match_probability': row['match_probability'],
                'is_true_match': None,  # To be filled manually
                'notes': ''
            })

        return {
            'sample': pd.DataFrame(validation_template),
            'instructions': (
                "1. Review each pair\n"
                "2. Set 'is_true_match' to True/False\n"
                "3. Add notes if needed\n"
                "4. Calculate: Precision = True Matches / Total"
            )
        }

    def identify_false_positive_patterns(self, manual_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze common patterns in false positives

        Args:
            manual_labels: DataFrame with columns [name_l, name_r, is_true_match]

        Returns:
            DataFrame with common false positive patterns
        """
        false_positives = manual_labels[manual_labels['is_true_match'] == False]

        patterns = []
        for _, row in false_positives.iterrows():
            pattern = self._identify_pattern(row['name_l'], row['name_r'])
            patterns.append(pattern)

        pattern_df = pd.DataFrame(patterns)
        pattern_summary = pattern_df.groupby('pattern_type').size().reset_index()
        pattern_summary.columns = ['pattern_type', 'count']

        return pattern_summary.sort_values('count', ascending=False)

    def _identify_pattern(self, name_l: str, name_r: str) -> Dict:
        """Identify the pattern causing false positive"""
        import re

        # Extract university names
        univ_pattern = r'of ([^,]+University[^,]*)|, ([^,]+University[^,]*)$|Affiliated to ([^,]+)$'
        univ_l = re.search(univ_pattern, name_l)
        univ_r = re.search(univ_pattern, name_r)

        univ_l = univ_l.group(1) or univ_l.group(2) or univ_l.group(3) if univ_l else None
        univ_r = univ_r.group(1) or univ_r.group(2) or univ_r.group(3) if univ_r else None

        # Check for hospital order
        order_pattern = r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth)'
        order_l = re.search(order_pattern, name_l)
        order_r = re.search(order_pattern, name_r)

        order_l = order_l.group(1) if order_l else None
        order_r = order_r.group(1) if order_r else None

        # Classify pattern
        if univ_l and univ_r and univ_l != univ_r:
            pattern_type = "different_university_same_hospital_type"
        elif order_l and order_r and order_l == order_r:
            pattern_type = "same_order_different_org"
        elif "Affiliated" in name_l and "Affiliated" in name_r:
            pattern_type = "generic_affiliated_hospital"
        else:
            pattern_type = "other"

        return {
            'pattern_type': pattern_type,
            'name_l': name_l,
            'name_r': name_r,
            'univ_l': univ_l,
            'univ_r': univ_r,
            'order_l': order_l,
            'order_r': order_r
        }

    def analyze_cluster_quality(self, metrics_df: pd.DataFrame) -> Dict:
        """
        Analyze cluster quality metrics

        Args:
            metrics_df: Cluster metrics from Splink

        Returns:
            Dictionary with quality statistics
        """
        return {
            'total_clusters': len(metrics_df),
            'avg_density': metrics_df['density'].mean(),
            'avg_centralization': metrics_df['cluster_centralisation'].mean(),
            'large_clusters': len(metrics_df[metrics_df['n_nodes'] >= 5]),
            'low_density_clusters': len(metrics_df[
                (metrics_df['density'] < 0.6) & (metrics_df['n_nodes'] >= 3)
            ]),
            'suspicious_clusters': len(metrics_df[
                (metrics_df['density'] < 0.5) & (metrics_df['n_nodes'] >= 3)
            ]),
            'size_distribution': metrics_df['n_nodes'].value_counts().to_dict()
        }

    def export_review_sample(self, output_path: str, n_samples: int = 30):
        """
        Export a sample of predictions for manual review

        Args:
            output_path: Path to save CSV
            n_samples: Number of predictions to sample
        """
        # Stratified sampling: high, medium, low confidence
        high_conf = self.predictions[self.predictions['match_probability'] >= 0.95].sample(
            min(10, len(self.predictions[self.predictions['match_probability'] >= 0.95])),
            random_state=42
        )
        med_conf = self.predictions[
            (self.predictions['match_probability'] >= 0.85) &
            (self.predictions['match_probability'] < 0.95)
        ].sample(
            min(10, len(self.predictions[
                (self.predictions['match_probability'] >= 0.85) &
                (self.predictions['match_probability'] < 0.95)
            ])),
            random_state=42
        )
        low_conf = self.predictions[
            (self.predictions['match_probability'] >= 0.75) &
            (self.predictions['match_probability'] < 0.85)
        ].sample(
            min(10, len(self.predictions[
                (self.predictions['match_probability'] >= 0.75) &
                (self.predictions['match_probability'] < 0.85)
            ])),
            random_state=42
        )

        sample = pd.concat([high_conf, med_conf, low_conf])

        review_df = sample[[
            'name_l', 'name_r', 'match_probability', 'match_weight'
        ]].copy()
        review_df['is_true_match'] = ''
        review_df['reviewer_notes'] = ''
        review_df['confidence_bucket'] = pd.cut(
            review_df['match_probability'],
            bins=[0, 0.85, 0.95, 1.0],
            labels=['Low (0.75-0.85)', 'Medium (0.85-0.95)', 'High (0.95+)']
        )

        review_df.to_csv(output_path, index=False)
        print(f"✓ Exported {len(review_df)} predictions for review to {output_path}")
        print("  Fill in 'is_true_match' column (TRUE/FALSE) and return for analysis")


class LabelGenerator:
    """Generate labeled training/test data"""

    @staticmethod
    def create_obvious_matches(df: pd.DataFrame) -> List[Dict]:
        """
        Create labeled pairs for obvious matches

        Args:
            df: Original dataframe with organization names

        Returns:
            List of labeled pairs
        """
        import re

        labeled_pairs = []

        # Pattern 1: Same org with "Affiliated to" vs "of" variation
        # e.g., "Hospital, University" vs "Hospital of University"
        for idx, row in df.iterrows():
            name = row['name']

            # Create variation
            if ', ' in name and 'University' in name:
                variation = name.replace(', ', ' of ')
                # Check if variation exists
                match = df[df['name'] == variation]
                if len(match) > 0:
                    labeled_pairs.append({
                        'name_l': name,
                        'name_r': variation,
                        'clerical_match_score': 1
                    })

        return labeled_pairs

    @staticmethod
    def create_obvious_non_matches(df: pd.DataFrame) -> List[Dict]:
        """
        Create labeled pairs for obvious non-matches

        Args:
            df: Original dataframe with organization names

        Returns:
            List of labeled pairs
        """
        import re

        labeled_pairs = []

        # Pattern: Same hospital type (e.g., "First Affiliated") but different universities
        first_affiliated = df[df['name'].str.contains('First Affiliated Hospital of', na=False)]

        # Sample pairs with different universities
        if len(first_affiliated) >= 2:
            for i in range(min(5, len(first_affiliated))):
                for j in range(i+1, min(i+3, len(first_affiliated))):
                    row_i = first_affiliated.iloc[i]
                    row_j = first_affiliated.iloc[j]

                    # Extract university names
                    univ_i = re.search(r'of ([^,]+)$', row_i['name'])
                    univ_j = re.search(r'of ([^,]+)$', row_j['name'])

                    if univ_i and univ_j and univ_i.group(1) != univ_j.group(1):
                        labeled_pairs.append({
                            'name_l': row_i['name'],
                            'name_r': row_j['name'],
                            'clerical_match_score': 0
                        })

        return labeled_pairs


class ThresholdOptimizer:
    """Optimize match threshold for precision/recall trade-off"""

    def __init__(self, predictions_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Initialize optimizer

        Args:
            predictions_df: All pairwise predictions
            labels_df: Labeled pairs with true match status
        """
        self.predictions = predictions_df
        self.labels = labels_df

    def calculate_metrics_at_threshold(self, threshold: float) -> Dict:
        """
        Calculate precision, recall, F1 at a given threshold

        Args:
            threshold: Match probability threshold

        Returns:
            Dictionary with metrics
        """
        # Merge predictions with labels
        merged = self.predictions.merge(
            self.labels,
            left_on=['unique_id_l', 'unique_id_r'],
            right_on=['unique_id_l', 'unique_id_r'],
            how='inner'
        )

        if len(merged) == 0:
            return {
                'threshold': threshold,
                'error': 'No overlap between predictions and labels'
            }

        # Apply threshold
        merged['predicted_match'] = merged['match_probability'] >= threshold

        # Calculate metrics
        tp = len(merged[(merged['predicted_match'] == True) & (merged['clerical_match_score'] == 1)])
        fp = len(merged[(merged['predicted_match'] == True) & (merged['clerical_match_score'] == 0)])
        tn = len(merged[(merged['predicted_match'] == False) & (merged['clerical_match_score'] == 0)])
        fn = len(merged[(merged['predicted_match'] == False) & (merged['clerical_match_score'] == 1)])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

    def find_optimal_threshold(
        self,
        thresholds: List[float] = None,
        optimize_for: str = 'f1'
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold

        Args:
            thresholds: List of thresholds to test (default: 0.70 to 0.99)
            optimize_for: 'f1', 'precision', or 'recall'

        Returns:
            Tuple of (best_threshold, metrics_at_best_threshold)
        """
        if thresholds is None:
            thresholds = [i/100 for i in range(70, 100, 2)]

        results = []
        for t in thresholds:
            metrics = self.calculate_metrics_at_threshold(t)
            results.append(metrics)

        results_df = pd.DataFrame(results)

        if optimize_for not in results_df.columns:
            raise ValueError(f"optimize_for must be one of: {results_df.columns.tolist()}")

        best_idx = results_df[optimize_for].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_metrics = results_df.loc[best_idx].to_dict()

        return best_threshold, best_metrics, results_df


def compare_model_versions(
    version1_predictions: pd.DataFrame,
    version2_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    version1_name: str = "Version 1",
    version2_name: str = "Version 2"
) -> pd.DataFrame:
    """
    Compare two model versions

    Args:
        version1_predictions: Predictions from first model
        version2_predictions: Predictions from second model
        labels: Ground truth labels
        version1_name: Name for first version
        version2_name: Name for second version

    Returns:
        Comparison DataFrame
    """
    opt1 = ThresholdOptimizer(version1_predictions, labels)
    opt2 = ThresholdOptimizer(version2_predictions, labels)

    # Calculate at standard threshold
    metrics1 = opt1.calculate_metrics_at_threshold(0.85)
    metrics2 = opt2.calculate_metrics_at_threshold(0.85)

    comparison = pd.DataFrame([
        {
            'Model': version1_name,
            'Precision': metrics1['precision'],
            'Recall': metrics1['recall'],
            'F1': metrics1['f1'],
            'TP': metrics1['tp'],
            'FP': metrics1['fp']
        },
        {
            'Model': version2_name,
            'Precision': metrics2['precision'],
            'Recall': metrics2['recall'],
            'F1': metrics2['f1'],
            'TP': metrics2['tp'],
            'FP': metrics2['fp']
        }
    ])

    return comparison


if __name__ == "__main__":
    print("Evaluation Tools for Splink Organization Matching")
    print("\nUsage:")
    print("  from evaluation_tools import MatchEvaluator, LabelGenerator, ThresholdOptimizer")
    print("\nExample:")
    print("  evaluator = MatchEvaluator(predictions_df, clusters_df)")
    print("  sample = evaluator.calculate_precision_from_sample(sample_size=20)")
    print("  evaluator.export_review_sample('review_sample.csv', n_samples=30)")
