"""
Advanced Blocking and Performance Optimization for Splink
==========================================================

This module provides production-ready patterns for:
- Multi-column blocking rules
- Salting for load balancing
- Blocking rule optimization
- Performance tuning

Compatible with Splink 3.9.11+
"""

from typing import List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np


# =============================================================================
# BLOCKING RULE GENERATOR
# =============================================================================

class BlockingRuleGenerator:
    """Helper class for generating optimized blocking rules"""

    @staticmethod
    def exact_match_rule(col: str) -> str:
        """Simple exact match blocking rule"""
        return f"l.{col} = r.{col}"

    @staticmethod
    def multi_column_exact_match(cols: List[str]) -> str:
        """Exact match on multiple columns (AND)"""
        conditions = [f"l.{col} = r.{col}" for col in cols]
        return " AND ".join(conditions)

    @staticmethod
    def prefix_match_rule(col: str, length: int) -> str:
        """First N characters match"""
        return f"substr(l.{col}, 1, {length}) = substr(r.{col}, 1, {length})"

    @staticmethod
    def prefix_and_exact_rule(prefix_col: str, prefix_len: int, exact_col: str) -> str:
        """Prefix match on one column + exact match on another"""
        return f"substr(l.{prefix_col}, 1, {prefix_len}) = substr(r.{prefix_col}, 1, {prefix_len}) AND l.{exact_col} = r.{exact_col}"

    @staticmethod
    def fuzzy_within_exact_rule(fuzzy_col: str, fuzzy_threshold: int, exact_col: str) -> str:
        """
        Fuzzy match on one column within exact match on another.
        Example: Similar names within same city.
        """
        return f"levenshtein(l.{fuzzy_col}, r.{fuzzy_col}) <= {fuzzy_threshold} AND l.{exact_col} = r.{exact_col}"

    @staticmethod
    def date_range_rule(date_col: str, days: int) -> str:
        """Dates within N days"""
        return f"abs(datediff('day', l.{date_col}, r.{date_col})) <= {days}"

    @staticmethod
    def array_intersection_rule(array_col: str, min_overlap: int = 1) -> str:
        """At least N elements in common"""
        return f"array_length(array_intersect(l.{array_col}, r.{array_col})) >= {min_overlap}"

    @staticmethod
    def null_safe_exact_match(col: str) -> str:
        """Exact match that only triggers when both sides are non-null"""
        return f"l.{col} = r.{col} AND l.{col} IS NOT NULL AND r.{col} IS NOT NULL"


# =============================================================================
# SALTING STRATEGIES
# =============================================================================

class SaltingStrategy:
    """Helper class for configuring salting"""

    @staticmethod
    def create_salted_rule(
        blocking_rule: str,
        num_partitions: int
    ) -> Dict[str, Union[str, int]]:
        """
        Create a salted blocking rule for load balancing.

        Args:
            blocking_rule: SQL blocking rule
            num_partitions: Number of partitions (higher = more parallelism)

        Returns:
            Dictionary with blocking_rule and salting_partitions
        """
        return {
            "blocking_rule": blocking_rule,
            "salting_partitions": num_partitions
        }

    @staticmethod
    def estimate_salting_partitions(
        df: pd.DataFrame,
        blocking_col: str,
        target_comparisons_per_partition: int = 1_000_000
    ) -> int:
        """
        Estimate optimal number of salting partitions.

        Args:
            df: Input dataframe
            blocking_col: Column used in blocking rule
            target_comparisons_per_partition: Target comparisons per partition

        Returns:
            Recommended number of partitions
        """
        # Count records per blocking key value
        value_counts = df[blocking_col].value_counts()

        # Estimate comparisons
        # For each blocking key value with n records, we get n*(n-1)/2 comparisons
        total_comparisons = (value_counts * (value_counts - 1) / 2).sum()

        # Calculate partitions needed
        partitions = int(np.ceil(total_comparisons / target_comparisons_per_partition))

        # Clamp to reasonable range
        partitions = max(1, min(partitions, 200))

        return partitions

    @staticmethod
    def get_adaptive_salting_config(
        blocking_rules: List[str],
        df: pd.DataFrame,
        blocking_cols: List[str]
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Create blocking rules with adaptive salting based on data skew.

        Args:
            blocking_rules: List of SQL blocking rules
            df: Input dataframe
            blocking_cols: Corresponding blocking columns for each rule

        Returns:
            List of blocking rules (some salted, some not)
        """
        if len(blocking_rules) != len(blocking_cols):
            raise ValueError("blocking_rules and blocking_cols must be same length")

        result = []

        for rule, col in zip(blocking_rules, blocking_cols):
            # Check skew
            value_counts = df[col].value_counts()
            max_count = value_counts.max()
            median_count = value_counts.median()

            # If highly skewed, use salting
            if max_count > median_count * 10:
                partitions = SaltingStrategy.estimate_salting_partitions(df, col)
                result.append(SaltingStrategy.create_salted_rule(rule, partitions))
                print(f"Salting '{col}' with {partitions} partitions (skew detected)")
            else:
                result.append(rule)
                print(f"No salting for '{col}' (low skew)")

        return result


# =============================================================================
# BLOCKING OPTIMIZATION
# =============================================================================

class BlockingOptimizer:
    """Helper class for optimizing blocking rules"""

    @staticmethod
    def analyze_blocking_rule_coverage(
        df: pd.DataFrame,
        blocking_rules: List[str],
        blocking_cols: List[str]
    ) -> pd.DataFrame:
        """
        Analyze how many record pairs each blocking rule would generate.

        Args:
            df: Input dataframe
            blocking_rules: List of SQL blocking rules (simplified)
            blocking_cols: Corresponding columns for analysis

        Returns:
            DataFrame with analysis results
        """
        results = []

        for rule, col in zip(blocking_rules, blocking_cols):
            value_counts = df[col].value_counts()

            # Estimate comparisons
            comparisons = (value_counts * (value_counts - 1) / 2).sum()

            # Estimate unique pairs
            total_possible_pairs = len(df) * (len(df) - 1) / 2

            results.append({
                'blocking_rule': rule,
                'blocking_column': col,
                'estimated_comparisons': int(comparisons),
                'pct_of_total_pairs': round(100 * comparisons / total_possible_pairs, 2),
                'num_distinct_keys': len(value_counts),
                'max_records_per_key': int(value_counts.max()),
                'median_records_per_key': int(value_counts.median()),
                'skew_ratio': round(value_counts.max() / value_counts.median(), 2)
            })

        return pd.DataFrame(results).sort_values('estimated_comparisons', ascending=False)

    @staticmethod
    def suggest_blocking_rules(
        df: pd.DataFrame,
        candidate_columns: List[str],
        max_rules: int = 5
    ) -> List[str]:
        """
        Suggest blocking rules based on data characteristics.

        Args:
            df: Input dataframe
            candidate_columns: Columns to consider for blocking
            max_rules: Maximum number of rules to suggest

        Returns:
            List of suggested blocking rules
        """
        suggestions = []

        for col in candidate_columns:
            # Skip columns with too many nulls
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > 0.5:
                print(f"Skipping {col}: {null_pct*100:.1f}% nulls")
                continue

            # Calculate selectivity
            distinct_count = df[col].nunique()
            selectivity = distinct_count / len(df)

            # Good blocking columns have moderate selectivity (not too unique, not too common)
            if 0.01 < selectivity < 0.8:
                suggestions.append({
                    'column': col,
                    'selectivity': selectivity,
                    'distinct_values': distinct_count,
                    'null_pct': null_pct
                })

        # Sort by selectivity (prefer moderate selectivity)
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df['selectivity_score'] = abs(suggestions_df['selectivity'] - 0.1)  # Prefer ~10%
        suggestions_df = suggestions_df.sort_values('selectivity_score')

        # Generate rules
        rules = []
        for _, row in suggestions_df.head(max_rules).iterrows():
            col = row['column']
            if row['selectivity'] < 0.05:
                # Low selectivity → exact match
                rules.append(f"l.{col} = r.{col}")
            else:
                # Higher selectivity → prefix match
                rules.append(f"substr(l.{col}, 1, 10) = substr(r.{col}, 1, 10)")

        return rules


# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

class PerformanceTuner:
    """Helper class for performance optimization"""

    @staticmethod
    def get_training_vs_prediction_blocking_rules(
        base_rules: List[str],
        training_multiplier: float = 0.5
    ) -> tuple:
        """
        Create separate blocking rules for training vs prediction.

        Training can use tighter rules (faster), prediction should be comprehensive.

        Args:
            base_rules: Base blocking rules
            training_multiplier: Factor to make training rules tighter

        Returns:
            (training_rules, prediction_rules)
        """
        # For training, use tighter rules
        training_rules = []
        for rule in base_rules:
            # If it's a prefix rule, increase length
            if 'substr(' in rule:
                # Extract length
                import re
                match = re.search(r'substr\(.*?,\s*1,\s*(\d+)\)', rule)
                if match:
                    length = int(match.group(1))
                    new_length = int(length * (1 + training_multiplier))
                    new_rule = rule.replace(
                        f"substr(l.{match.group(0).split('.')[1].split(',')[0]}, 1, {length})",
                        f"substr(l.{match.group(0).split('.')[1].split(',')[0]}, 1, {new_length})"
                    )
                    training_rules.append(new_rule)
                else:
                    training_rules.append(rule)
            else:
                training_rules.append(rule)

        # For prediction, use original (broader) rules
        prediction_rules = base_rules

        return training_rules, prediction_rules

    @staticmethod
    def estimate_runtime(
        num_records: int,
        blocking_rules: List[str],
        blocking_cols: List[str],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Estimate runtime for linkage job.

        Args:
            num_records: Number of records
            blocking_rules: Blocking rules
            blocking_cols: Blocking columns
            df: Sample dataframe for analysis

        Returns:
            Dictionary with runtime estimates
        """
        # Estimate comparisons
        total_comparisons = 0

        for col in blocking_cols:
            value_counts = df[col].value_counts()
            comparisons = (value_counts * (value_counts - 1) / 2).sum()
            total_comparisons += comparisons

        # Rough runtime estimates (based on DuckDB benchmarks)
        # Assumes ~500k comparisons/second on modern hardware
        comparisons_per_second = 500_000

        estimated_seconds = total_comparisons / comparisons_per_second

        return {
            'total_comparisons': int(total_comparisons),
            'estimated_seconds': round(estimated_seconds, 2),
            'estimated_minutes': round(estimated_seconds / 60, 2),
            'comparisons_per_second': comparisons_per_second,
            'recommendation': (
                "Fast (< 1 min)" if estimated_seconds < 60 else
                "Moderate (1-10 min)" if estimated_seconds < 600 else
                "Slow (> 10 min) - consider tighter blocking or Spark"
            )
        }


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

class BlockingConfigs:
    """Pre-built blocking configurations for common scenarios"""

    @staticmethod
    def person_matching_rules() -> List[Union[str, Dict[str, Any]]]:
        """Blocking rules for person/customer matching"""
        return [
            # High confidence: exact email
            "l.email = r.email AND l.email IS NOT NULL",

            # Name + DOB
            "l.surname = r.surname AND l.dob = r.dob",

            # Phone
            "l.phone = r.phone AND l.phone IS NOT NULL",

            # Name prefix + postcode
            "substr(l.surname, 1, 5) = substr(r.surname, 1, 5) AND l.postcode = r.postcode",

            # Name + city (for records without postcode)
            "l.surname = r.surname AND l.city = r.city",
        ]

    @staticmethod
    def organization_matching_rules() -> List[Union[str, Dict[str, Any]]]:
        """Blocking rules for organization matching"""
        return [
            # Core company name (from feature engineering)
            "l.core_company_name = r.core_company_name",

            # Same parent company
            "l.parent_company = r.parent_company AND l.parent_company IS NOT NULL",

            # Name prefix + country
            "substr(l.core_company_name, 1, 10) = substr(r.core_company_name, 1, 10) AND l.country_code = r.country_code",

            # Website domain
            "l.website_domain = r.website_domain AND l.website_domain IS NOT NULL",

            # Address for headquarters
            "l.headquarters_address_normalized = r.headquarters_address_normalized AND l.headquarters_address_normalized IS NOT NULL",
        ]

    @staticmethod
    def hospital_university_matching_rules() -> List[Union[str, Dict[str, Any]]]:
        """
        Blocking rules for hospitals/universities.

        Critical: Always block on university to prevent false positives.
        """
        return [
            # Same university + hospital type
            "l.university_name = r.university_name AND l.hospital_type = r.hospital_type AND l.university_name IS NOT NULL",

            # Same university only (broader)
            "l.university_name = r.university_name AND l.university_name IS NOT NULL",

            # Name prefix (but require long prefix to avoid FPs)
            "substr(l.name, 1, 25) = substr(r.name, 1, 25)",

            # City + hospital type (for non-university hospitals)
            "l.city = r.city AND l.hospital_type = r.hospital_type AND l.university_name IS NULL AND r.university_name IS NULL",
        ]

    @staticmethod
    def address_matching_rules() -> List[Union[str, Dict[str, Any]]]:
        """Blocking rules for address matching"""
        return [
            # Postcode
            "l.postcode = r.postcode",

            # Street name + city
            "l.street_name_normalized = r.street_name_normalized AND l.city = r.city",

            # Building number + postcode
            "l.building_number = r.building_number AND l.postcode = r.postcode",

            # Geographic distance (if lat/lon available)
            # This would require custom SQL with haversine formula
        ]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Advanced Blocking Optimization for Splink")
    print("=" * 50)

    # Example dataset
    df = pd.DataFrame({
        'name': ['John Smith'] * 1000 + ['Jane Doe'] * 500 + list(range(500)),
        'city': ['New York'] * 500 + ['Boston'] * 500 + ['Other'] * 1000,
        'dob': pd.date_range('1950-01-01', periods=2000)
    })

    print("\n1. ANALYZE BLOCKING RULES")
    optimizer = BlockingOptimizer()

    blocking_rules = [
        "l.name = r.name",
        "l.city = r.city",
        "l.dob = r.dob"
    ]
    blocking_cols = ['name', 'city', 'dob']

    analysis = optimizer.analyze_blocking_rule_coverage(
        df, blocking_rules, blocking_cols
    )
    print(analysis)

    print("\n2. ESTIMATE SALTING PARTITIONS")
    salting = SaltingStrategy()

    # Name is skewed (many "John Smith")
    name_partitions = salting.estimate_salting_partitions(df, 'name')
    print(f"Recommended partitions for 'name': {name_partitions}")

    # City is less skewed
    city_partitions = salting.estimate_salting_partitions(df, 'city')
    print(f"Recommended partitions for 'city': {city_partitions}")

    print("\n3. ADAPTIVE SALTING")
    salted_rules = salting.get_adaptive_salting_config(
        blocking_rules, df, blocking_cols
    )
    print("Salted rules:")
    for rule in salted_rules:
        print(f"  {rule}")

    print("\n4. PRE-BUILT CONFIGURATIONS")
    print("\nPerson matching rules:")
    for rule in BlockingConfigs.person_matching_rules()[:3]:
        print(f"  - {rule}")

    print("\nOrganization matching rules:")
    for rule in BlockingConfigs.organization_matching_rules()[:3]:
        print(f"  - {rule}")

    print("\n" + "=" * 50)
    print("See ADVANCED_SPLINK_TECHNIQUES.md for full documentation")
