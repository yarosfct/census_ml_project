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
├── notebooks/               # Jupyter notebooks for exploration
│   └── README.md
├── reports/                 # Generated analysis reports
│   ├── figures/             # Generated graphics and figures
│   ├── tables/              # Generated tables
│   └── README.md
├── src/                     # Source code
│   ├── census_ml/           # Main package
│   │   ├── config.py        # Configuration and constants
│   │   ├── data/            # Data loading utilities
│   │   ├── features/        # Feature engineering and preprocessing
│   │   ├── models/          # Model definitions and utilities
│   │   ├── eval/            # Evaluation and cross-validation
│   │   └── utils/           # General utilities (logging, etc.)
│   └── scripts/             # Standalone scripts
│       ├── quick_check.py   # Verify package setup
│       └── run_experiments.py  # Main experiment runner (to be implemented)
├── tests/                   # Test suite
│   └── test_imports.py      # Test package integrity
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

4. **Verify installation**

```bash
# Using make
make quickcheck

# Or manually
python src/scripts/quick_check.py
```

5. **Run tests**

```bash
# Using make
make test

# Or manually
pytest
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

## Development

### Code Quality

```bash
# Format code
make format  # Uses ruff format

# Lint code
make lint    # Uses ruff check

# Run tests
make test    # Uses pytest
```

### Makefile Commands

- `make venv` - Create virtual environment
- `make install` - Install dependencies
- `make test` - Run test suite
- `make lint` - Run linter
- `make format` - Format code
- `make quickcheck` - Run quick verification script
- `make clean` - Remove generated files

## Reproducibility

This project follows best practices for reproducible machine learning research:

- **Random seed**: All random operations use `RANDOM_SEED=42` (configurable in `src/census_ml/config.py`)
- **Dependency pinning**: All dependencies are specified in `pyproject.toml`
- **Environment isolation**: Use virtual environments
- **Data versioning**: Raw data is immutable; all transformations are scripted
- **Results tracking**: Experimental results will be saved in `reports/`

## Data Policy

**Important**: No raw data is committed to this repository. The `data/raw/`, `data/interim/`, and `data/processed/` directories are ignored by git. Only code and documentation are version-controlled.

Please download the dataset separately and place it in the appropriate directory (see instructions above).

## Project Phases

- **✓ Step 0** (Current): Project structure and boilerplate
- **Step 1**: Data loading and exploratory analysis
- **Step 2**: Baseline preprocessing pipeline
- **Step 3**: Enhanced preprocessing pipeline
- **Step 4**: Model training and evaluation
- **Step 5**: Statistical comparison and reporting

## Contributors

- Your Name ([@yarosfct](https://github.com/yarosfct))

## License

MIT License - see LICENSE file for details

## References

- Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/adult

## Step 0 Completion Checklist

This project has completed the initial scaffolding phase (Step 0):

- [x] Git repository initialized and remote configured
- [x] Folder structure created
- [x] Configuration files (.gitignore, .editorconfig, pyproject.toml, Makefile)
- [x] Python package structure (src/census_ml/)
- [x] Core modules with minimal boilerplate
- [x] Test suite setup
- [x] Documentation (README files)
- [x] Quick check script passes
- [x] Import tests pass
- [x] Code formatting and linting configured

### Verification

Run these commands to verify the setup:

```bash
# Check imports and package structure
python src/scripts/quick_check.py

# Run test suite
pytest

# Check code quality
ruff check .
```

All checks should pass without errors.
