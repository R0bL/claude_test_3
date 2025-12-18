# Corporate Entity Matching Strategy

## The Challenge: Pfizer Example

When matching corporate entities like Pfizer, you face a unique problem: **hierarchical organization structures** where you need to decide what "match" means.

### What Should vs Shouldn't Match?

```
✓ SHOULD MATCH (Same Entity):
- "Pfizer Inc" ↔ "Pfizer Inc."
- "Pfizer Inc, 235 East 42nd Street, New York, NY 10017" ↔ "Pfizer Inc 235 East 42nd Street,New York,NY10017"
- "PFIZER INC" ↔ "Pfizer Inc."

✗ SHOULD NOT MATCH (Different Entities, Same Parent):
- "Pfizer (Austria)" ↔ "Pfizer (Germany)"
- "Pfizer (United States)" ↔ "Pfizer (Canada)"

? DEPENDS ON USE CASE:
- "Medivation LLC, a wholly owned subsidiary of Pfizer" ↔ "Pfizer Inc"
  → Same parent, different legal entities
```

## Core Strategy Overview

### 1. Extract Hierarchical Features

Instead of just matching on names, extract:

```python
# From: "Pfizer (Austria)"
{
    'core_company_name': 'pfizer',
    'country_code_from_name': 'AT',
    'parent_company': None,
    'entity_type': 'regional_office'
}

# From: "Medivation LLC, a wholly owned subsidiary of Pfizer Inc."
{
    'core_company_name': 'medivation',
    'country_code_from_name': None,
    'parent_company': 'pfizer',
    'entity_type': 'subsidiary'
}

# From: "Pfizer Inc, 235 East 42nd Street, New York, NY 10017"
{
    'core_company_name': 'pfizer',
    'normalized_address': '235 e 42nd st new york ny 10017',
    'parent_company': None,
    'entity_type': 'parent'
}
```

### 2. Define Match Levels

You need TWO types of matching:

#### **Level 1: Entity-Level Matching** (What you probably want)
Match only when it's the **exact same legal entity**:
- "Pfizer (Austria)" matches only other "Pfizer (Austria)" records
- "Pfizer Inc" matches only other "Pfizer Inc" records
- "Pfizer (Austria)" does NOT match "Pfizer (Germany)"

#### **Level 2: Parent-Level Matching** (For relationship mapping)
Link entities that share the same parent:
- Create clusters showing all Pfizer subsidiaries
- Build organization hierarchies
- Map corporate structure

## Implementation Details

### Feature Extraction

#### 1. Core Company Name

Remove noise to get the base company:

```
"Pfizer (Austria)" → "pfizer"
"Pfizer Inc, 235 East 42nd Street..." → "pfizer"
"Pfizer Clinical Research Unit" → "pfizer"
"PFIZER INC." → "pfizer"
```

#### 2. Country Code

Extract from parenthetical notation:

```
"Pfizer (Austria)" → "AT"
"Pfizer (Germany)" → "DE"
"Pfizer Inc" → None
```

#### 3. Parent Company

Detect subsidiary relationships:

```
"Medivation LLC, a wholly owned subsidiary of Pfizer Inc." → "pfizer"
"Array BioPharma, now a wholly owned subsidiary of Pfizer" → "pfizer"
"Hospira, now a wholly owned subsidiary of Pfizer" → "pfizer"
```

#### 4. Address Normalization

Standardize address variations:

```
"235 East 42nd Street, New York, NY 10017" →
"235 e 42nd st new york ny 10017"

"235 East 42nd Street,New York,NY10017" →
"235 e 42nd st new york ny 10017"
```

#### 5. Entity Type Classification

Categorize organizations:

| Type | Example | Matching Rule |
|------|---------|---------------|
| `regional_office` | "Pfizer (Germany)" | Match only same country |
| `subsidiary` | "Medivation, a subsidiary of Pfizer" | Match by subsidiary name |
| `research_unit` | "Pfizer Clinical Research Unit" | Match by unit location/name |
| `parent` | "Pfizer Inc" | Match by core name + address |
| `collaboration` | "Bristol Myers Squibb and Pfizer" | Often exclude from matching |

### Blocking Rules for Corporate Matching

Use these specific blocking rules:

```python
blocking_rules = [
    # Rule 1: Exact core company name
    "l.core_company_name = r.core_company_name",

    # Rule 2: Same parent company (for subsidiaries)
    "l.parent_company = r.parent_company AND l.parent_company IS NOT NULL",

    # Rule 3: Core name + country (for regional offices)
    "substr(l.core_company_name, 1, 10) = substr(r.core_company_name, 1, 10) "
    "AND l.country_code_from_name = r.country_code_from_name",

    # Rule 4: Same address (for HQ variations)
    "l.normalized_address = r.normalized_address "
    "AND l.normalized_address IS NOT NULL",
]
```

### Comparison Strategy

#### For Regional Offices

```python
# Compare: "Pfizer (Austria)" vs "Pfizer (Germany)"

if core_name_l == core_name_r:  # Both "pfizer"
    if country_l == country_r:
        → MATCH (same entity)
    else:
        → NO MATCH (different regional offices)
```

#### For Subsidiaries

```python
# Compare: "Medivation, a subsidiary of Pfizer" vs "Pfizer Inc"

if parent_l == 'pfizer' and core_r == 'pfizer':
    → RELATED but NO MATCH (subsidiary vs parent)

if parent_l == 'pfizer' and parent_r == 'pfizer':
    if core_l == core_r:
        → MATCH (same subsidiary)
    else:
        → NO MATCH (different subsidiaries, same parent)
```

#### For Address Variations

```python
# Compare: "Pfizer Inc, 235 East 42nd Street..." vs "Pfizer Inc 235 E 42nd St..."

if core_name_l == core_name_r:  # Both "pfizer"
    if normalized_address_l == normalized_address_r:
        → MATCH (same HQ, formatting differences)
    elif both addresses contain "235" and "42nd" and "new york":
        → HIGH CONFIDENCE MATCH
```

## Post-Processing Rules

After Splink generates predictions, apply filters:

### Rule 1: Exclude Regional Office Cross-Matches

```python
# Filter out: "Pfizer (Austria)" ↔ "Pfizer (Germany)"
if (entity_type_l == 'regional_office' and
    entity_type_r == 'regional_office' and
    country_l != country_r):
    REJECT
```

### Rule 2: Exclude Parent-Subsidiary Matches

```python
# Filter out: "Medivation, a subsidiary of Pfizer" ↔ "Pfizer Inc"
if (parent_l == core_r or parent_r == core_l):
    REJECT  # They're related but not the same entity
```

### Rule 3: Exclude Collaborations

```python
# Filter out: "Bristol Myers Squibb and Pfizer"
if is_collaboration:
    REJECT  # Multi-company entities
```

## Building Organization Hierarchies

For advanced use cases, create a hierarchy:

```
Pfizer Inc (Parent)
├── Pfizer (Austria) - Regional Office
├── Pfizer (Germany) - Regional Office
├── Pfizer (United States) - Regional Office
├── Medivation LLC - Wholly Owned Subsidiary
├── Array BioPharma - Wholly Owned Subsidiary
├── Hospira - Wholly Owned Subsidiary
└── Pfizer Clinical Research Unit - Brussels - Research Facility
```

### Implementation

```python
def create_hierarchy(matches_df):
    # 1. Identify parent companies (entity_type == 'parent')
    # 2. Link subsidiaries to parents (via parent_company field)
    # 3. Link regional offices to parents (via core_company_name)
    # 4. Create tree structure

    return {
        'parent': 'Pfizer Inc',
        'subsidiaries': [
            {'name': 'Medivation LLC', 'type': 'wholly_owned'},
            {'name': 'Array BioPharma', 'type': 'wholly_owned'}
        ],
        'regional_offices': [
            {'name': 'Pfizer (Austria)', 'country': 'AT'},
            {'name': 'Pfizer (Germany)', 'country': 'DE'}
        ]
    }
```

## Comparison Matrix

| Pair Type | Should Match? | Key Differentiator |
|-----------|---------------|-------------------|
| Same parent HQ, different addresses | ✓ Yes | `normalized_address` similarity |
| Same company, different countries | ✗ No | `country_code_from_name` |
| Subsidiary vs Parent | ✗ No | `parent_company` vs `core_company_name` |
| Different subsidiaries, same parent | ✗ No | `core_company_name` differs |
| Same subsidiary, different records | ✓ Yes | `core_company_name` + `parent_company` |
| Collaboration entity | ✗ No | `is_collaboration` flag |

## Example Workflow

### Input Data
```
1. Pfizer (Austria)
2. Pfizer (Germany)
3. Pfizer Inc
4. Pfizer Inc, 235 East 42nd Street, New York, NY 10017
5. PFIZER INC.
6. Medivation LLC, a wholly owned subsidiary of Pfizer Inc.
7. Pfizer Clinical Research Unit - Brussels
```

### After Feature Extraction
```
1. core=pfizer, country=AT, type=regional_office
2. core=pfizer, country=DE, type=regional_office
3. core=pfizer, country=None, type=parent
4. core=pfizer, country=None, type=parent, address=235 e 42nd st...
5. core=pfizer, country=None, type=parent
6. core=medivation, parent=pfizer, type=subsidiary
7. core=pfizer, country=None, type=research_unit
```

### After Entity-Level Matching
```
Cluster 1: {3, 4, 5}  # All "Pfizer Inc" variations
Cluster 2: {1}        # Pfizer (Austria) - unique
Cluster 3: {2}        # Pfizer (Germany) - unique
Cluster 4: {6}        # Medivation - unique (subsidiary)
Cluster 5: {7}        # Pfizer CRU - unique (research unit)
```

### With Parent-Level Grouping (Optional)
```
Pfizer Corporate Family:
  - Cluster 1: Pfizer Inc (parent) {3, 4, 5}
  - Cluster 2: Pfizer (Austria) {1}
  - Cluster 3: Pfizer (Germany) {2}
  - Cluster 4: Medivation LLC (subsidiary) {6}
  - Cluster 5: Pfizer CRU Brussels {7}
```

## Key Differences from Hospital Matching

| Aspect | Hospital Matching | Corporate Matching |
|--------|-------------------|-------------------|
| **Main Challenge** | Name variations (affiliated vs of) | Hierarchical structure |
| **Geographic Role** | Differentiator within same system | Defines separate entities |
| **Key Features** | University name, hospital order | Parent company, country, address |
| **False Positive Risk** | Similar names, different universities | Regional offices, subsidiaries |
| **Blocking Strategy** | Name prefix + university | Core name + country/address |
| **Post-Processing** | Filter low-density clusters | Filter cross-country, parent-child |

## Recommended Splink Configuration

```python
from splink import SettingsCreator
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_level_library as cll

settings = SettingsCreator(
    link_type="dedupe_only",

    comparisons=[
        # Core company name - PRIMARY
        cl.CustomComparison(
            output_column_name="core_company_name",
            comparison_levels=[
                cll.NullLevel("core_company_name"),
                cll.ExactMatchLevel("core_company_name",
                                   term_frequency_adjustments=True),
                cll.JaroWinklerLevel("core_company_name", 0.95),
                cll.JaroWinklerLevel("core_company_name", 0.90),
                cll.ElseLevel()
            ]
        ),

        # Country code - CRITICAL for regional offices
        cl.ExactMatch("country_code_from_name").configure(
            m_probabilities=[0.95, 0.05],  # High weight when matched
        ),

        # Parent company - for subsidiaries
        cl.ExactMatch("parent_company").configure(
            term_frequency_adjustments=True
        ),

        # Normalized address - for HQ variations
        cl.ExactMatch("normalized_address"),

        # Entity type - ensure we're comparing similar types
        cl.ExactMatch("entity_type"),
    ],

    blocking_rules_to_generate_predictions=[
        "l.core_company_name = r.core_company_name",
        "l.parent_company = r.parent_company AND l.parent_company IS NOT NULL",
        "substr(l.core_company_name, 1, 10) = substr(r.core_company_name, 1, 10) "
        "AND l.country_code_from_name = r.country_code_from_name",
    ]
)
```

## Special Considerations

### 1. Acquisitions & Historical Names

```
"Pharmacia" was acquired by Pfizer in 2003
"Wyeth" was acquired by Pfizer in 2009

Should "Pharmacia" match "Pfizer"?
→ Depends on time period of data
→ Consider adding "historical_parent" field
→ Or create separate "acquisition_mapping" table
```

### 2. Joint Ventures

```
"Pfizer-BioNTech"
"Pfizer and BioNTech"

These are collaborations, not subsidiaries
→ Mark as entity_type='collaboration'
→ Extract both company names
→ Create separate records for each?
```

### 3. Research Units

```
"Pfizer Clinical Research Unit - Brussels"
"Belgium Pfizer Clinical Research Unit"

Same or different?
→ Extract location
→ Compare: parent + location + unit type
```

## Next Steps

1. **Prepare your data** using `prepare_corporate_data()`
2. **Configure Splink** with corporate-specific comparisons
3. **Apply blocking rules** that account for hierarchies
4. **Post-process** to filter regional/subsidiary cross-matches
5. **Create hierarchies** (optional) for visualization

## When to Use What

| Use Case | Match Level | Post-Processing |
|----------|-------------|-----------------|
| Deduplicate company records | Entity-level | Filter regional + subsidiaries |
| Find all Pfizer-related entities | Parent-level | Keep all relationships |
| Clean master data | Entity-level | Aggressive filtering |
| Build org charts | Both | Create hierarchy trees |
| Merge financial data | Entity-level | Very strict (country required) |

---

**Bottom Line**: Corporate matching requires understanding **organizational structure**, not just name similarity. You must decide whether "Pfizer (Germany)" and "Pfizer (France)" are "the same" for your use case.
