# Adult Census Income Dataset - Summary Report

## Overall Dataset Statistics

- **Total Rows**: 48,842
- **Total Columns**: 15
- **Features**: 14
  - Numerical: 6
  - Categorical: 8

### Missing Values

- **Total Missing Values**: 6,465
- **Columns with Missing Values**:

  - `fnlwgt`: 2,799 (5.73%)
  - `workclass`: 2,809 (5.75%)
  - `native_country`: 857 (1.75%)

### Target Distribution

- `<=50K`: 37,155 (76.07%)
- `>50K`: 11,687 (23.93%)

## Dataset Variants

Two dataset variants have been prepared for different modeling approaches:

### Variant: DROP

- **Strategy**: Drop
- **Rows Kept**: 45,222
- **Rows Dropped**: 3,620
- **Missing Values Remaining**: 0

**Class Balance**:
- Class 0 (<=50K): 34,014 (75.22%)
- Class 1 (>50K): 11,208 (24.78%)

**Note**: All rows with missing values removed. Ready for modeling.

### Variant: IMPUTE

- **Strategy**: Impute
- **Rows Kept**: 48,842
- **Rows Dropped**: 0
- **Missing Values Remaining**: 6,465

**Class Balance**:
- Class 0 (<=50K): 37,155 (76.07%)
- Class 1 (>50K): 11,687 (23.93%)

**Columns with Missing Values**: `fnlwgt`, `workclass`, `native_country`

**Note**: Missing values preserved as NaN. Must use SimpleImputer in sklearn Pipeline to avoid leakage.

## Leakage Prevention

To prevent data leakage, preprocessing follows these principles:

1. **IMPUTE variant**: Missing values are preserved as NaN. Imputation must be performed inside an sklearn Pipeline during cross-validation, ensuring imputation parameters are fitted only on training folds.

2. **DROP variant**: Complete-case analysis. All rows with missing values removed. No imputation needed, but sample size is reduced.

3. **No preprocessing fitted on full data**: This inspection phase only examines the data; no preprocessing transformations have been fitted that could leak information from test sets.

## Generated Artifacts

### Tables (CSV)
- `tables/missingness.csv` - Missing value statistics
- `tables/target_distribution.csv` - Target class distribution
- `tables/categorical_cardinality.csv` - Unique values per categorical feature
- `tables/numeric_stats.csv` - Descriptive statistics for numeric features

### Figures (PNG)
- `figures/missingness_percentage.png`
- `figures/target_distribution.png`
- `figures/age_histogram.png`
- `figures/hours_per_week_histogram.png`
- `figures/capital_gain_histogram.png`
- `figures/capital_loss_histogram.png`

