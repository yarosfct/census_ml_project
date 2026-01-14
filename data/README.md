# Data Directory

This directory contains all data used in the Adult Census Income classification project.

## Directory Structure

- `raw/` - Raw, immutable data as downloaded from the source
- `interim/` - Intermediate data that has been transformed
- `processed/` - Final processed data ready for modeling

## Data Policy

**Important**: The contents of this directory (except this README) are **not tracked by git**. Raw data files should never be committed to version control.

## Adult Census Income Dataset

### Source

- **Dataset**: Adult Census Income
- **Repository**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult
- **Citation**: Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository.

### Download Instructions

Download the dataset files and place them in `data/raw/`:

```bash
cd data/raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

### Dataset Description

The Adult dataset contains census data extracted from the 1994 Census database. The prediction task is to determine whether a person's income exceeds $50K/year based on demographic and employment attributes.

**Instances**: 48,842 (32,561 in training set, 16,281 in test set)

**Features**: 14 attributes

#### Numerical Features (6)
- `age`: Age in years
- `fnlwgt`: Final weight (sampling weight)
- `education_num`: Number of years of education
- `capital_gain`: Capital gains
- `capital_loss`: Capital losses
- `hours_per_week`: Hours worked per week

#### Categorical Features (8)
- `workclass`: Employment type (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education`: Highest education level (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `marital_status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Occupation type (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Male, Female)
- `native_country`: Country of origin

#### Target Variable
- `income`: Income class (<=50K, >50K)

### Missing Values

Some instances have missing values indicated by a question mark (`?`). Missing values appear in:
- `workclass`
- `occupation`
- `native_country`

### Data Characteristics

- **Class imbalance**: The dataset is imbalanced with approximately 75% of individuals earning ≤50K
- **Mixed types**: Contains both numerical and categorical features
- **Real-world data**: Contains typical data quality issues (missing values, class imbalance)

## Usage

Data loading utilities are provided in `src/census_ml/data/load_data.py`. See the module documentation for details.

## Notes

- The test set file (`adult.test`) has a different format than the training set (includes a period after the class label)
- Duplicate records exist in the dataset and should be handled during preprocessing
- Some categorical features have many unique values, requiring careful encoding strategies
