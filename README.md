# Adult Census Income Classification

A machine learning project to predict whether an individual's annual income exceeds $50,000 using demographic and employment-related attributes from the UCI Adult Census Income dataset.

## Project Goal

This project aims to evaluate how different preprocessing strategies and hyperparameter tuning affect the performance of classical machine-learning algorithms on the Adult Census Income classification task. The primary research question is:

> *How much do preprocessing choices and hyperparameter tuning influence the predictive performance of classical machine-learning models on the Adult income prediction task?*

## Repository Structure

```
.
├── data/                    # Data directory (contents not tracked)
│   ├── raw/                 # Raw, immutable data
│   ├── interim/             # Intermediate transformed data
│   ├── processed/           # Final preprocessed data ready for modeling
│   └── README.md            # Data documentation and instructions
├── docs/                    # Documentation and literature
│   ├── Literature/          # Reference papers
│   └── ML_First_Milestone_Report.md
├── reports/                 # Generated analysis reports
│   ├── figures/             # Generated graphics and figures
│   ├── results/             # Generated results
│   ├── tables/              # Generated tables
│   └── dataset_summary.md   # Dataset analysis summary
├── src/                     # Source code
│   ├── census_ml/           # Main package
│   │   ├── config.py        # Configuration and constants
│   │   ├── data/            # Data loading utilities
│   │   ├── fairness/        # Fairness analysis
│   │   ├── features/        # Preprocessing
│   │   ├── models/          # Model definitions and utilities
│   │   ├── eval/            # Evaluation and cross-validation
│   │   └── utils/           # General utilities (logging, etc.)
│   └── scripts/             # Standalone scripts
│       ├── inspect_and_prepare.py   # Dataset analysis
│       └── run_experiments.py  # Main experiment runner (to be implemented)
├── .editorconfig            # Editor configuration
├── .gitignore               # Git ignore rules
├── Makefile                 # Automation commands
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

## Quickstart

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yarosfct/census_ml_project.git
cd census_ml_project
```

2. **Create a virtual environment**

```bash
# Using make
make venv

# Or manually
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

3. **Install dependencies**

```bash
# Using make
make install

# Or manually
pip install --upgrade pip
pip install -e ".[dev]"
```


### Download the Dataset

The Adult Census Income dataset is not included in this repository. Download it from the UCI Machine Learning Repository:

- **Dataset URL**: https://archive.ics.uci.edu/ml/datasets/adult
- **Direct links**:
  - Training data: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
  - Test data: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

Place the downloaded files in `data/raw/`:

```bash
cd data/raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
cd ../..
```

See `data/README.md` for more information about the dataset.

### Running the experiments

After downloading the dataset, run the experiments:

```bash
python -m src.scripts.run_experiments
```



## Reproducibility

- **Random seed**: All random operations use `RANDOM_SEED=42` (configurable in `src/census_ml/config.py`)
- **Dependency pinning**: All dependencies are specified in `pyproject.toml`
- **Environment isolation**: Use virtual environments
- **Data versioning**: Raw data is immutable; all transformations are scripted
- **Results tracking**: Experimental results will be saved in `reports/`


## References

- Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/adult
