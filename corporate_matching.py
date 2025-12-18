"""
Corporate Entity Matching Extensions for Splink

This module extends the base organization matching with specific
features for matching multinational corporations and their subsidiaries.
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple


class CorporateEntityExtractor:
    """Extract features specific to corporate entities"""

    # Known parent companies and their variations
    PARENT_COMPANIES = {
        'pfizer': ['pfizer inc', 'pfizer', 'pfizer incorporated'],
        'novartis': ['novartis', 'novartis ag'],
        'roche': ['roche', 'f. hoffmann-la roche'],
        # Add more as needed
    }

    # Subsidiary indicators
    SUBSIDIARY_PATTERNS = [
        r'wholly owned subsidiary of (.+?)(?:,|$|\.|Inc)',
        r'subsidiary of (.+?)(?:,|$|\.|Inc)',
        r'a (.+?) company',
        r'division of (.+?)(?:,|$)',
        r'div of (.+?)(?:,|$)',
    ]

    def extract_country_code(self, name: str) -> Optional[str]:
        """
        Extract country code from organization name

        Examples:
        - "Pfizer (Austria)" -> "AT"
        - "Pfizer (Germany)" -> "DE"
        """
        # Pattern: "Name (Country)"
        country_pattern = r'\(([A-Za-z\s]+)\)$'
        match = re.search(country_pattern, name)

        if match:
            country_name = match.group(1).strip()
            return self._country_name_to_code(country_name)

        return None

    def _country_name_to_code(self, country_name: str) -> Optional[str]:
        """Map country name to ISO code"""
        country_map = {
            'austria': 'AT',
            'germany': 'DE',
            'ireland': 'IE',
            'spain': 'ES',
            'sweden': 'SE',
            'india': 'IN',
            'australia': 'AU',
            'south korea': 'KR',
            'united kingdom': 'GB',
            'italy': 'IT',
            'netherlands': 'NL',
            'belgium': 'BE',
            'france': 'FR',
            'united states': 'US',
            'canada': 'CA',
            'switzerland': 'CH',
            'japan': 'JP',
            'china': 'CN',
            'thailand': 'TH',
            # Add more as needed
        }
        return country_map.get(country_name.lower())

    def extract_parent_company(self, name: str) -> Optional[str]:
        """
        Extract parent company name from organization name

        Examples:
        - "Medivation LLC, a wholly owned subsidiary of Pfizer Inc." -> "pfizer"
        - "Array BioPharma Inc. (a wholly owned subsidiary of Pfizer Inc.)" -> "pfizer"
        """
        for pattern in self.SUBSIDIARY_PATTERNS:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                parent = match.group(1).strip().lower()
                # Normalize to known parent
                for canonical, variations in self.PARENT_COMPANIES.items():
                    if any(var in parent for var in variations):
                        return canonical
                return parent

        return None

    def extract_core_company_name(self, name: str) -> str:
        """
        Extract core company name without country/subsidiary info

        Examples:
        - "Pfizer (Austria)" -> "pfizer"
        - "Pfizer Inc, 235 East 42nd Street, New York, NY 10017" -> "pfizer"
        - "Pfizer Clinical Research Unit" -> "pfizer"
        """
        # Remove country codes in parentheses
        name = re.sub(r'\s*\([A-Za-z\s]+\)\s*$', '', name)

        # Remove addresses (anything with street numbers)
        name = re.sub(r',?\s*\d+\s+[A-Z][a-z]+.*', '', name)

        # Remove subsidiary indicators
        for pattern in self.SUBSIDIARY_PATTERNS:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)

        # Remove common suffixes
        suffixes = [
            r'\s+Inc\.?$',
            r'\s+LLC\.?$',
            r'\s+Ltd\.?$',
            r'\s+Limited\.?$',
            r'\s+Corporation\.?$',
            r'\s+Corp\.?$',
            r'\s+GmbH\.?$',
            r'\s+AG\.?$',
            r'\s+S\.A\.?$',
            r'\s+Pty\.?$',
        ]

        for suffix in suffixes:
            name = re.sub(suffix, '', name, flags=re.IGNORECASE)

        # Remove division/unit indicators
        name = re.sub(r'\s*-\s*(Clinical Research Unit|Research Unit|Division).*', '', name, flags=re.IGNORECASE)

        return name.strip().lower()

    def is_collaboration(self, name: str) -> bool:
        """
        Check if organization name represents a collaboration

        Examples:
        - "Bristol Myers Squibb and Pfizer" -> True
        - "Pfizer Inc" -> False
        """
        collaboration_indicators = [' and ', ' & ', ',']

        # Check if multiple company names
        for indicator in collaboration_indicators:
            if indicator in name:
                parts = re.split(r'\s+(?:and|&|,)\s+', name, flags=re.IGNORECASE)
                # If we have multiple parts that look like company names
                if len(parts) >= 2:
                    return True

        return False

    def normalize_address(self, name: str) -> Optional[str]:
        """
        Normalize address component of organization name

        Examples:
        - "235 East 42nd Street, New York, NY 10017" -> "235 e 42nd st new york ny 10017"
        - "235 East 42nd Street,New York,NY 10017" -> "235 e 42nd st new york ny 10017"
        """
        # Extract potential address
        address_pattern = r'(\d+\s+[A-Z][a-z]+\s+\d+[a-z]{2}\s+[Ss]treet.*?(?:NY|New York).*?\d{5})'
        match = re.search(address_pattern, name)

        if not match:
            return None

        address = match.group(1)

        # Normalize
        address = address.lower()
        address = re.sub(r'\s+', ' ', address)  # Normalize whitespace
        address = re.sub(r',\s*', ' ', address)  # Remove commas
        address = re.sub(r'\bstreet\b', 'st', address)
        address = re.sub(r'\beastern?\b', 'e', address)
        address = re.sub(r'\bwestern?\b', 'w', address)
        address = re.sub(r'\bnorthern?\b', 'n', address)
        address = re.sub(r'\bsouthern?\b', 's', address)

        return address.strip()

    def determine_entity_type(self, name: str) -> str:
        """
        Classify entity type

        Returns:
        - "regional_office": e.g., "Pfizer (Germany)"
        - "subsidiary": e.g., "Medivation, a subsidiary of Pfizer"
        - "research_unit": e.g., "Pfizer Clinical Research Unit"
        - "collaboration": e.g., "Pfizer and Novartis"
        - "parent": e.g., "Pfizer Inc"
        """
        if self.is_collaboration(name):
            return "collaboration"

        if self.extract_parent_company(name):
            return "subsidiary"

        if re.search(r'\([A-Z][a-z]+\)$', name):
            return "regional_office"

        if re.search(r'(clinical research|research unit|cru)\b', name, re.IGNORECASE):
            return "research_unit"

        return "parent"


class CorporateMatchingStrategy:
    """Define matching strategies for different entity types"""

    @staticmethod
    def should_match_regional_offices(name1: str, name2: str,
                                     country1: str, country2: str,
                                     core1: str, core2: str) -> str:
        """
        Determine if two regional offices should match

        Returns:
        - "same_entity": Same regional office
        - "same_parent": Different regional offices, same parent
        - "different": Different companies
        """
        # Same core name
        if core1 == core2:
            # Same country
            if country1 == country2:
                return "same_entity"
            # Different countries
            elif country1 and country2 and country1 != country2:
                return "same_parent"

        return "different"

    @staticmethod
    def should_match_subsidiaries(name1: str, name2: str,
                                 parent1: Optional[str], parent2: Optional[str],
                                 core1: str, core2: str) -> str:
        """
        Determine if two subsidiaries should match

        Returns:
        - "same_entity": Same subsidiary
        - "same_parent": Different subsidiaries, same parent
        - "different": Different companies
        """
        # Both are subsidiaries of the same parent
        if parent1 and parent2 and parent1 == parent2:
            # Check if same subsidiary name
            if core1 == core2:
                return "same_entity"
            else:
                return "same_parent"

        # Only one is a subsidiary
        if parent1 or parent2:
            # Check if the subsidiary's parent matches the other's core
            if parent1 and parent1 == core2:
                return "same_parent"
            if parent2 and parent2 == core1:
                return "same_parent"

        return "different"


def prepare_corporate_data(df: pd.DataFrame, name_column: str = 'name') -> pd.DataFrame:
    """
    Prepare corporate data with extracted features

    Args:
        df: Input dataframe
        name_column: Column containing organization names

    Returns:
        DataFrame with additional feature columns
    """
    extractor = CorporateEntityExtractor()

    df = df.copy()

    # Extract features
    df['country_code_from_name'] = df[name_column].apply(extractor.extract_country_code)
    df['parent_company'] = df[name_column].apply(extractor.extract_parent_company)
    df['core_company_name'] = df[name_column].apply(extractor.extract_core_company_name)
    df['normalized_address'] = df[name_column].apply(extractor.normalize_address)
    df['entity_type'] = df[name_column].apply(extractor.determine_entity_type)
    df['is_collaboration'] = df[name_column].apply(extractor.is_collaboration)

    return df


def create_corporate_blocking_rules() -> List[str]:
    """
    Create blocking rules specific to corporate matching

    Returns:
        List of SQL blocking rule strings
    """
    return [
        # Block 1: Same core company name (exact)
        "l.core_company_name = r.core_company_name",

        # Block 2: Same parent company
        "l.parent_company = r.parent_company AND l.parent_company IS NOT NULL",

        # Block 3: Same core name prefix + same country
        "substr(l.core_company_name, 1, 10) = substr(r.core_company_name, 1, 10) AND l.country_code_from_name = r.country_code_from_name",

        # Block 4: Same address (for US headquarters with different formats)
        "l.normalized_address = r.normalized_address AND l.normalized_address IS NOT NULL",

        # Block 5: Core name fuzzy match (using soundex or similar)
        "substr(l.core_company_name, 1, 8) = substr(r.core_company_name, 1, 8)",
    ]


def post_process_corporate_matches(
    predictions_df: pd.DataFrame,
    data_df: pd.DataFrame,
    match_level: str = "entity"
) -> pd.DataFrame:
    """
    Post-process matches to apply corporate-specific rules

    Args:
        predictions_df: Splink predictions
        data_df: Original data with extracted features
        match_level: "entity" (exact match) or "parent" (include same-parent matches)

    Returns:
        Filtered predictions
    """
    # Merge with original data to get features
    predictions_with_features = predictions_df.merge(
        data_df[['unique_id', 'core_company_name', 'country_code_from_name',
                 'parent_company', 'entity_type', 'is_collaboration']],
        left_on='unique_id_l',
        right_on='unique_id',
        how='left',
        suffixes=('', '_l')
    ).merge(
        data_df[['unique_id', 'core_company_name', 'country_code_from_name',
                 'parent_company', 'entity_type', 'is_collaboration']],
        left_on='unique_id_r',
        right_on='unique_id',
        how='left',
        suffixes=('_l', '_r')
    )

    strategy = CorporateMatchingStrategy()

    # Filter based on match level
    if match_level == "entity":
        # Remove matches where it's just same parent but different subsidiaries
        mask = ~(
            (predictions_with_features['parent_company_l'] == predictions_with_features['parent_company_r']) &
            (predictions_with_features['parent_company_l'].notna()) &
            (predictions_with_features['core_company_name_l'] != predictions_with_features['core_company_name_r'])
        )

        # Remove matches where regional offices differ by country
        mask = mask & ~(
            (predictions_with_features['entity_type_l'] == 'regional_office') &
            (predictions_with_features['entity_type_r'] == 'regional_office') &
            (predictions_with_features['country_code_from_name_l'] != predictions_with_features['country_code_from_name_r']) &
            (predictions_with_features['country_code_from_name_l'].notna()) &
            (predictions_with_features['country_code_from_name_r'].notna())
        )

        predictions_with_features = predictions_with_features[mask]

    # Remove collaboration entities
    predictions_with_features = predictions_with_features[
        ~predictions_with_features['is_collaboration_l'] &
        ~predictions_with_features['is_collaboration_r']
    ]

    return predictions_with_features


def create_entity_hierarchy(clusters_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical structure showing parent-subsidiary relationships

    Args:
        clusters_df: Clustered results from Splink
        data_df: Original data with extracted features

    Returns:
        DataFrame with hierarchy information
    """
    # Merge clusters with features
    hierarchy = clusters_df.merge(
        data_df[['unique_id', 'name', 'core_company_name', 'parent_company',
                 'entity_type', 'country_code_from_name']],
        on='unique_id',
        how='left'
    )

    # Group by cluster and identify parent
    def identify_cluster_parent(group):
        # Parent is the entity with entity_type='parent' or most common core name
        parents = group[group['entity_type'] == 'parent']
        if len(parents) > 0:
            return parents.iloc[0]['core_company_name']

        # Otherwise, use most common core name
        return group['core_company_name'].mode().iloc[0] if len(group) > 0 else None

    hierarchy['cluster_parent'] = hierarchy.groupby('cluster_id')['core_company_name'].transform(
        lambda x: x.mode().iloc[0] if len(x) > 0 else None
    )

    # Add relationship labels
    def label_relationship(row):
        if row['entity_type'] == 'parent':
            return 'parent_company'
        elif row['entity_type'] == 'regional_office':
            return f"regional_office_{row['country_code_from_name']}"
        elif row['entity_type'] == 'subsidiary':
            return 'subsidiary'
        elif row['entity_type'] == 'research_unit':
            return 'research_unit'
        else:
            return 'other'

    hierarchy['relationship'] = hierarchy.apply(label_relationship, axis=1)

    return hierarchy


if __name__ == "__main__":
    print("Corporate Entity Matching Extensions")
    print("\nUsage:")
    print("  from corporate_matching import prepare_corporate_data, create_corporate_blocking_rules")
    print("\nExample:")
    print("  df_prepared = prepare_corporate_data(df, name_column='name')")
    print("  blocking_rules = create_corporate_blocking_rules()")
