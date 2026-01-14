# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, experimentation, and visualization.

## Purpose

Notebooks are useful for:
- Exploratory data analysis (EDA)
- Visualizing data distributions and relationships
- Prototyping preprocessing pipelines
- Testing model performance interactively
- Creating visualizations for reports

## Organization

Notebooks should follow a naming convention for clarity:

- `01_eda_<description>.ipynb` - Exploratory data analysis
- `02_preprocessing_<description>.ipynb` - Preprocessing experiments
- `03_modeling_<description>.ipynb` - Model training and evaluation
- `04_analysis_<description>.ipynb` - Results analysis and visualization

## Best Practices

1. **Use descriptive names**: Make it clear what each notebook does
2. **Add markdown documentation**: Explain your thought process and findings
3. **Keep notebooks focused**: One notebook per specific task or analysis
4. **Clear outputs before committing**: Run "Restart & Clear Output" before committing to keep diffs clean
5. **Move production code to modules**: Once code is stable, refactor it into `src/census_ml/`

## Running Notebooks

Make sure you have installed the project dependencies:

```bash
pip install -e ".[dev]"
pip install jupyter  # If not already installed
```

Then start Jupyter:

```bash
jupyter notebook
```

Or use JupyterLab:

```bash
pip install jupyterlab
jupyter lab
```

## Importing the Package

To use the `census_ml` package in notebooks, you can import it directly:

```python
import sys
from pathlib import Path

# Add src to path (if needed)
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root / "src"))

# Now you can import
from census_ml import RANDOM_SEED
from census_ml.data.load_data import load_raw_data
from census_ml.models.model_zoo import get_baseline_models
```

Or install the package in editable mode (recommended):

```bash
pip install -e .
```

Then simply:

```python
from census_ml import RANDOM_SEED
from census_ml.data.load_data import load_raw_data
```
