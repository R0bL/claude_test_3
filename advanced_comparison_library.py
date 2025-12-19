"""
Advanced Comparison Functions for Splink
==========================================

This module extends Splink's comparison capabilities with production-ready
patterns for:
- Phonetic matching
- Geographic distance
- Fuzzy date matching
- Array/alias comparisons
- Token-based similarity

Compatible with Splink 3.9.11+
"""

import re
from typing import List, Optional, Dict, Any
import pandas as pd


# =============================================================================
# PHONETIC MATCHING
# =============================================================================

class PhoneticComparisons:
    """
    Helper class for creating phonetic comparison levels.

    Requires: pip install phonetics
    """

    @staticmethod
    def prepare_phonetic_columns(df: pd.DataFrame, name_column: str) -> pd.DataFrame:
        """
        Add phonetic encoding columns to dataframe.

        Args:
            df: Input dataframe
            name_column: Column containing names

        Returns:
            DataFrame with phonetic columns added
        """
        try:
            from phonetics import dmetaphone, soundex, metaphone
        except ImportError:
            raise ImportError(
                "phonetics library required. Install with: pip install phonetics"
            )

        df = df.copy()
        df[f'{name_column}_dmeta'] = df[name_column].apply(
            lambda x: dmetaphone(str(x))[0] if pd.notna(x) else None
        )
        df[f'{name_column}_soundex'] = df[name_column].apply(
            lambda x: soundex(str(x)) if pd.notna(x) else None
        )
        df[f'{name_column}_metaphone'] = df[name_column].apply(
            lambda x: metaphone(str(x)) if pd.notna(x) else None
        )

        return df

    @staticmethod
    def get_phonetic_comparison_dict(
        col_name: str,
        include_dmeta: bool = True,
        include_soundex: bool = False,
        include_metaphone: bool = False
    ) -> Dict[str, Any]:
        """
        Create a custom comparison with phonetic levels.

        Returns a comparison dictionary that can be used in Splink settings.

        Example:
            comparison = PhoneticComparisons.get_phonetic_comparison_dict("name")
            settings["comparisons"].append(comparison)
        """
        comparison_levels = [
            {
                "sql_condition": f"({col_name}_l IS NULL OR {col_name}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            },
            {
                "sql_condition": f"{col_name}_l = {col_name}_r",
                "label_for_charts": "Exact match",
                "tf_adjustment_column": col_name,
                "tf_adjustment_weight": 1.0
            }
        ]

        if include_dmeta:
            comparison_levels.append({
                "sql_condition": f"{col_name}_dmeta_l = {col_name}_dmeta_r",
                "label_for_charts": "Double Metaphone match"
            })

        if include_soundex:
            comparison_levels.append({
                "sql_condition": f"{col_name}_soundex_l = {col_name}_soundex_r",
                "label_for_charts": "Soundex match"
            })

        if include_metaphone:
            comparison_levels.append({
                "sql_condition": f"{col_name}_metaphone_l = {col_name}_metaphone_r",
                "label_for_charts": "Metaphone match"
            })

        # Fuzzy string matching
        comparison_levels.extend([
            {
                "sql_condition": f"jaro_winkler_similarity({col_name}_l, {col_name}_r) >= 0.95",
                "label_for_charts": "Jaro-Winkler >= 0.95",
                "tf_adjustment_column": col_name,
                "tf_adjustment_weight": 1.0
            },
            {
                "sql_condition": f"jaro_winkler_similarity({col_name}_l, {col_name}_r) >= 0.90",
                "label_for_charts": "Jaro-Winkler >= 0.90"
            },
            {
                "sql_condition": "ELSE",
                "label_for_charts": "All other comparisons"
            }
        ])

        return {
            "output_column_name": col_name,
            "comparison_description": f"{col_name} with phonetic matching",
            "comparison_levels": comparison_levels
        }


# =============================================================================
# GEOGRAPHIC DISTANCE
# =============================================================================

class GeographicComparisons:
    """Helper class for geographic distance comparisons"""

    @staticmethod
    def get_distance_comparison_dict(
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        km_thresholds: List[float] = [1, 10, 50, 100]
    ) -> Dict[str, Any]:
        """
        Create geographic distance comparison.

        Args:
            lat_col: Latitude column name
            lon_col: Longitude column name
            km_thresholds: Distance thresholds in kilometers

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({lat_col}_l IS NULL OR {lat_col}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            }
        ]

        for threshold in sorted(km_thresholds):
            comparison_levels.append({
                "sql_condition": f"""
                    6371 * acos(
                        cos(radians({lat_col}_l)) *
                        cos(radians({lat_col}_r)) *
                        cos(radians({lon_col}_r) - radians({lon_col}_l)) +
                        sin(radians({lat_col}_l)) *
                        sin(radians({lat_col}_r))
                    ) <= {threshold}
                """,
                "label_for_charts": f"Within {threshold} km"
            })

        comparison_levels.append({
            "sql_condition": "ELSE",
            "label_for_charts": f"All other comparisons"
        })

        return {
            "output_column_name": "location",
            "comparison_description": "Geographic distance",
            "comparison_levels": comparison_levels
        }

    @staticmethod
    def geocode_postcode_to_lat_lon(df: pd.DataFrame, postcode_col: str) -> pd.DataFrame:
        """
        Add latitude/longitude from postcode.

        Note: This is a stub. In production, use a geocoding service like:
        - geopy
        - pgeocode
        - Google Maps API
        """
        # TODO: Implement actual geocoding
        # For now, return df unchanged
        print("Warning: Geocoding not implemented. Add lat/lon manually or use geocoding service.")
        return df


# =============================================================================
# FUZZY DATE MATCHING
# =============================================================================

class DateComparisons:
    """Helper class for fuzzy date comparisons"""

    @staticmethod
    def get_date_comparison_dict(
        date_col: str,
        thresholds_days: List[int] = [0, 1, 30, 365],
        allow_year_only: bool = True
    ) -> Dict[str, Any]:
        """
        Create fuzzy date comparison.

        Args:
            date_col: Date column name
            thresholds_days: Thresholds in days [exact, 1 day, 1 month, 1 year]
            allow_year_only: Include year-only match level

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({date_col}_l IS NULL OR {date_col}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            }
        ]

        for threshold in sorted(thresholds_days):
            if threshold == 0:
                label = "Exact match"
                sql_condition = f"{date_col}_l = {date_col}_r"
            else:
                label = f"Within {threshold} days"
                sql_condition = f"abs(datediff('day', {date_col}_l, {date_col}_r)) <= {threshold}"

            comparison_levels.append({
                "sql_condition": sql_condition,
                "label_for_charts": label
            })

        if allow_year_only:
            comparison_levels.append({
                "sql_condition": f"extract(year from {date_col}_l) = extract(year from {date_col}_r)",
                "label_for_charts": "Same year only"
            })

        comparison_levels.append({
            "sql_condition": "ELSE",
            "label_for_charts": "All other comparisons"
        })

        return {
            "output_column_name": date_col,
            "comparison_description": f"Fuzzy {date_col} comparison",
            "comparison_levels": comparison_levels
        }


# =============================================================================
# ARRAY/ALIAS COMPARISONS
# =============================================================================

class ArrayComparisons:
    """Helper class for comparing arrays/lists (e.g., aliases)"""

    @staticmethod
    def prepare_alias_columns(
        df: pd.DataFrame,
        alias_columns: List[str],
        output_col: str = "all_names"
    ) -> pd.DataFrame:
        """
        Combine multiple name/alias columns into array.

        Args:
            df: Input dataframe
            alias_columns: List of columns containing name variations
            output_col: Name for output array column

        Returns:
            DataFrame with array column added
        """
        df = df.copy()

        def combine_names(row):
            names = []
            for col in alias_columns:
                if pd.notna(row.get(col)):
                    names.append(str(row[col]))
            return names if names else None

        df[output_col] = df.apply(combine_names, axis=1)
        return df

    @staticmethod
    def get_array_exact_comparison_dict(
        array_col: str,
        min_intersections: List[int] = [3, 1]
    ) -> Dict[str, Any]:
        """
        Create exact array intersection comparison.

        Args:
            array_col: Column containing arrays
            min_intersections: Minimum intersection sizes to check

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({array_col}_l IS NULL OR {array_col}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            }
        ]

        for min_int in sorted(min_intersections, reverse=True):
            comparison_levels.append({
                "sql_condition": f"array_length(array_intersect({array_col}_l, {array_col}_r)) >= {min_int}",
                "label_for_charts": f"{min_int}+ aliases match"
            })

        comparison_levels.append({
            "sql_condition": "ELSE",
            "label_for_charts": "All other comparisons"
        })

        return {
            "output_column_name": array_col,
            "comparison_description": f"Array intersection on {array_col}",
            "comparison_levels": comparison_levels
        }

    @staticmethod
    def get_array_fuzzy_comparison_dict(
        array_col: str,
        similarity_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Create fuzzy array comparison (custom SQL).

        Note: This uses EXISTS with UNNEST - supported in DuckDB and some SQL backends.

        Args:
            array_col: Column containing arrays
            similarity_threshold: Jaro-Winkler threshold

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({array_col}_l IS NULL OR {array_col}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            },
            {
                "sql_condition": f"""
                    EXISTS (
                        SELECT 1
                        FROM unnest({array_col}_l) AS la
                        CROSS JOIN unnest({array_col}_r) AS ra
                        WHERE jaro_winkler_similarity(la, ra) >= {similarity_threshold}
                    )
                """,
                "label_for_charts": f"Any alias fuzzy match >= {similarity_threshold}"
            },
            {
                "sql_condition": "ELSE",
                "label_for_charts": "All other comparisons"
            }
        ]

        return {
            "output_column_name": f"{array_col}_fuzzy",
            "comparison_description": f"Fuzzy array comparison on {array_col}",
            "comparison_levels": comparison_levels
        }


# =============================================================================
# TOKEN-BASED COMPARISONS (JACCARD)
# =============================================================================

class TokenComparisons:
    """Helper class for token-based comparisons (addresses, descriptions)"""

    @staticmethod
    def get_jaccard_comparison_dict(
        col_name: str,
        thresholds: List[float] = [0.9, 0.7, 0.5]
    ) -> Dict[str, Any]:
        """
        Create Jaccard similarity comparison.

        Best for multi-word strings like addresses.

        Args:
            col_name: Column name
            thresholds: Jaccard similarity thresholds

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({col_name}_l IS NULL OR {col_name}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            },
            {
                "sql_condition": f"{col_name}_l = {col_name}_r",
                "label_for_charts": "Exact match"
            }
        ]

        for threshold in sorted(thresholds, reverse=True):
            comparison_levels.append({
                "sql_condition": f"jaccard({col_name}_l, {col_name}_r) >= {threshold}",
                "label_for_charts": f"Jaccard >= {threshold}"
            })

        comparison_levels.append({
            "sql_condition": "ELSE",
            "label_for_charts": "All other comparisons"
        })

        return {
            "output_column_name": col_name,
            "comparison_description": f"Jaccard similarity on {col_name}",
            "comparison_levels": comparison_levels
        }


# =============================================================================
# COMPOSITE COMPARISONS (AND/OR LOGIC)
# =============================================================================

class CompositeComparisons:
    """Helper class for creating AND/OR composite comparisons"""

    @staticmethod
    def get_and_comparison_dict(
        col_names: List[str],
        label: str
    ) -> Dict[str, Any]:
        """
        Create comparison requiring ALL conditions.

        Example: Match on name AND address AND DOB

        Args:
            col_names: List of columns that must all match
            label: Label for this comparison level

        Returns:
            Comparison dictionary for Splink settings
        """
        conditions = [f"{col}_l = {col}_r" for col in col_names]
        sql_condition = " AND ".join(conditions)

        comparison_levels = [
            {
                "sql_condition": sql_condition,
                "label_for_charts": label
            },
            {
                "sql_condition": "ELSE",
                "label_for_charts": "All other comparisons"
            }
        ]

        return {
            "output_column_name": f"composite_{'_'.join(col_names)}",
            "comparison_description": f"Composite AND: {', '.join(col_names)}",
            "comparison_levels": comparison_levels
        }

    @staticmethod
    def get_hospital_university_comparison_dict(
        name_col: str = "name",
        university_col: str = "university_name",
        jw_thresholds: List[float] = [0.95, 0.90]
    ) -> Dict[str, Any]:
        """
        Hospital/university comparison requiring BOTH name match AND same university.

        This prevents false positives like:
        "First Hospital of X University" ≠ "First Hospital of Y University"

        Args:
            name_col: Hospital name column
            university_col: University name column
            jw_thresholds: Jaro-Winkler thresholds for name

        Returns:
            Comparison dictionary for Splink settings
        """
        comparison_levels = [
            {
                "sql_condition": f"({name_col}_l IS NULL OR {name_col}_r IS NULL)",
                "label_for_charts": "Null",
                "is_null_level": True
            },
            # Exact name + same university
            {
                "sql_condition": f"{name_col}_l = {name_col}_r AND {university_col}_l = {university_col}_r",
                "label_for_charts": "Exact name + same university"
            }
        ]

        # Fuzzy name + same university
        for threshold in sorted(jw_thresholds, reverse=True):
            comparison_levels.append({
                "sql_condition": f"""
                    jaro_winkler_similarity({name_col}_l, {name_col}_r) >= {threshold}
                    AND {university_col}_l = {university_col}_r
                """,
                "label_for_charts": f"Name JW>={threshold} + same university"
            })

        comparison_levels.append({
            "sql_condition": "ELSE",
            "label_for_charts": "All other comparisons"
        })

        return {
            "output_column_name": "hospital_with_university",
            "comparison_description": "Hospital name requiring university match",
            "comparison_levels": comparison_levels
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Advanced Comparison Library for Splink")
    print("=" * 50)

    # Example 1: Phonetic matching
    print("\n1. PHONETIC MATCHING")
    print("   - Prepare data:")
    print("     df = PhoneticComparisons.prepare_phonetic_columns(df, 'name')")
    print("   - Create comparison:")
    print("     comp = PhoneticComparisons.get_phonetic_comparison_dict('name')")

    # Example 2: Geographic distance
    print("\n2. GEOGRAPHIC DISTANCE")
    print("   - Create comparison:")
    print("     comp = GeographicComparisons.get_distance_comparison_dict(")
    print("         lat_col='latitude',")
    print("         lon_col='longitude',")
    print("         km_thresholds=[1, 10, 50]")
    print("     )")

    # Example 3: Fuzzy dates
    print("\n3. FUZZY DATE MATCHING")
    print("   - Create comparison:")
    print("     comp = DateComparisons.get_date_comparison_dict(")
    print("         date_col='date_of_birth',")
    print("         thresholds_days=[0, 1, 30, 365]")
    print("     )")

    # Example 4: Array/alias matching
    print("\n4. ARRAY/ALIAS MATCHING")
    print("   - Prepare data:")
    print("     df = ArrayComparisons.prepare_alias_columns(")
    print("         df, ['official_name', 'alias1', 'alias2'], 'all_names'")
    print("     )")
    print("   - Create comparison:")
    print("     comp = ArrayComparisons.get_array_fuzzy_comparison_dict('all_names')")

    # Example 5: Hospital/university matching
    print("\n5. HOSPITAL/UNIVERSITY MATCHING (prevents false positives)")
    print("   - Create comparison:")
    print("     comp = CompositeComparisons.get_hospital_university_comparison_dict(")
    print("         name_col='name',")
    print("         university_col='university_name'")
    print("     )")

    print("\n" + "=" * 50)
    print("See ADVANCED_SPLINK_TECHNIQUES.md for full documentation")
