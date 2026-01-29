"""
Script to generate poster-ready figures for the research poster.

This script creates publication-quality figures with appropriate sizing,
fonts, and styling for academic poster presentation.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy import stats

from census_ml.config import FIGURES_DIR, TARGET_COL, RESULTS_DIR
from census_ml.data.load_data import load_adult_dataset
from census_ml.utils.logging import get_logger

logger = get_logger(__name__)


def generate_class_distribution_figure(
    output_path: Path | None = None,
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 300,
) -> Path:
    """
    Generate a poster-ready class distribution bar chart.

    Args:
        output_path: Path to save the figure. If None, saves to FIGURES_DIR.
        figsize: Figure size in inches (width, height). Default: (10, 7).
        dpi: Resolution in dots per inch. Default: 300 (print quality).

    Returns:
        Path to the saved figure file.
    """
    logger.info("Loading dataset...")
    df = load_adult_dataset()

    logger.info("Calculating class distribution...")
    target_dist = df[TARGET_COL].value_counts()
    target_pct = (target_dist / len(df)) * 100

    # Create DataFrame for easier plotting
    target_df = pd.DataFrame(
        {
            "class": target_dist.index,
            "count": target_dist.values,
            "percentage": target_pct.values,
        }
    )

    # Sort by class name for consistent ordering
    target_df = target_df.sort_values("class")

    # Create figure with poster-appropriate styling
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Professional color scheme (blue and orange)
    colors = ["#4472C4", "#ED7D31"]

    # Create bar chart
    bars = ax.bar(
        target_df["class"],
        target_df["count"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        width=0.6,
    )

    # Add value labels on bars
    for i, (bar, cnt, pct) in enumerate(
        zip(bars, target_df["count"], target_df["percentage"], strict=True)
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Styling
    ax.set_xlabel("Income Class", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=18, fontweight="bold", pad=20)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Make left and bottom spines thicker
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add grid for easier reading (light gray)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Format y-axis to show thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = FIGURES_DIR / "poster_class_distribution.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved class distribution figure to {output_path}")
    return output_path


def generate_pipeline_flowchart(
    output_path: Path | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    dpi: int = 300,
) -> Path:
    """
    Generate a poster-ready pipeline flowchart diagram (vertical layout).

    Args:
        output_path: Path to save the figure. If None, saves to FIGURES_DIR.
        figsize: Figure size in inches (width, height). Default: (5.5, 4.5) - tall/narrow.
        dpi: Resolution in dots per inch. Default: 300 (print quality).

    Returns:
        Path to the saved figure file.
    """
    logger.info("Creating pipeline flowchart (vertical layout)...")

    # Create figure with vertical layout
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Vertical bounds - tall and narrow
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # Color scheme matching Figure 1
    colors = {
        "raw_data": "#D3D3D3",  # Light gray
        "preprocessing": "#4472C4",  # Blue
        "feature_selection": "#ED7D31",  # Orange
        "model": "#70AD47",  # Green
        "metrics": "#7030A0",  # Purple
    }

    # Box dimensions for vertical layout (use full width)
    box_width = 4.8
    box_height = 0.65
    box_spacing = 0.1
    x_center = 2.75  # Center of 5.5 width
    
    # Calculate y positions (top to bottom)
    y_positions = []
    current_y = 4.1  # Start from top
    for _ in range(5):  # 5 main boxes
        y_positions.append(current_y)
        current_y -= box_height + box_spacing

    from matplotlib.patches import Rectangle

    # Box 1: Raw Data
    box1 = FancyBboxPatch(
        (x_center - box_width / 2, y_positions[0] - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03",
        facecolor=colors["raw_data"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box1)
    ax.text(
        x_center,
        y_positions[0],
        "Raw Data\nN=48,842 | 14 features",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Arrow 1: Raw Data → Preprocessing (vertical, pointing down)
    arrow1 = FancyArrowPatch(
        (x_center, y_positions[0] - box_height / 2),
        (x_center, y_positions[1] + box_height / 2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.5,
        color="black",
    )
    ax.add_patch(arrow1)

    # Box 2: Preprocessing
    box2 = FancyBboxPatch(
        (x_center - box_width / 2, y_positions[1] - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03",
        facecolor=colors["preprocessing"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box2)
    ax.text(
        x_center,
        y_positions[1],
        "Preprocessing\nMissing→'Missing' | One-hot | StandardScaler",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    # Arrow 2: Preprocessing → Feature Selection (conditional)
    arrow2 = FancyArrowPatch(
        (x_center, y_positions[1] - box_height / 2),
        (x_center, y_positions[2] + box_height / 2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.5,
        color="black",
    )
    ax.add_patch(arrow2)

    # Box 3: Feature Selection (conditional, shown as optional)
    box3_bg = FancyBboxPatch(
        (x_center - box_width / 2, y_positions[2] - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03",
        facecolor=colors["feature_selection"],
        edgecolor="none",
        linewidth=0,
    )
    ax.add_patch(box3_bg)
    # Add dashed border
    box3_border = Rectangle(
        (x_center - box_width / 2, y_positions[2] - box_height / 2),
        box_width,
        box_height,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(box3_border)
    # Text for feature selection box
    ax.text(
        x_center,
        y_positions[2],
        "Feature Selection (SelectKBest)\n[Non-tree only]",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
        zorder=10,
    )

    # Show bypass path for tree models (side arrow)
    arrow_bypass = FancyArrowPatch(
        (x_center + box_width / 2 + 0.2, y_positions[1]),
        (x_center + box_width / 2 + 0.2, y_positions[3]),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.5,
        color="black",
        linestyle=":",
    )
    ax.add_patch(arrow_bypass)
    ax.text(
        x_center + box_width / 2 + 0.5,
        (y_positions[1] + y_positions[3]) / 2,
        "Tree\nmodels",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Arrow 3: Feature Selection → Model
    arrow3 = FancyArrowPatch(
        (x_center, y_positions[2] - box_height / 2),
        (x_center, y_positions[3] + box_height / 2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.5,
        color="black",
        linestyle="--",
    )
    ax.add_patch(arrow3)

    # Box 4: Model
    box4 = FancyBboxPatch(
        (x_center - box_width / 2, y_positions[3] - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03",
        facecolor=colors["model"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box4)
    ax.text(
        x_center,
        y_positions[3],
        "Model\nLogReg, Random Forest, XGBoost, etc.",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    # Arrow 4: Model → Metrics
    arrow4 = FancyArrowPatch(
        (x_center, y_positions[3] - box_height / 2),
        (x_center, y_positions[4] + box_height / 2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.5,
        color="black",
    )
    ax.add_patch(arrow4)

    # Box 5: Metrics
    box5 = FancyBboxPatch(
        (x_center - box_width / 2, y_positions[4] - box_height / 2),
        box_width,
        box_height,
        boxstyle="round,pad=0.03",
        facecolor=colors["metrics"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(box5)
    ax.text(
        x_center,
        y_positions[4],
        "Evaluation\nPrecision, Recall, F1, AUC",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    # Tight layout with minimal padding
    plt.tight_layout(pad=0.1)

    # Save figure with tight bounding box
    if output_path is None:
        output_path = FIGURES_DIR / "poster_pipeline_flowchart.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_path, 
        dpi=dpi, 
        bbox_inches="tight", 
        pad_inches=0.05,  # Minimal padding
        facecolor="white"
    )
    plt.close()

    logger.info(f"Saved pipeline flowchart to {output_path}")
    return output_path


def generate_nested_cv_diagram(
    output_path: Path | None = None,
    figsize: tuple[float, float] = (6, 2.4),
    dpi: int = 300,
) -> Path:
    """
    Generate a simple, clean nested cross-validation diagram for poster.

    Args:
        output_path: Path to save the figure. If None, saves to FIGURES_DIR.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch. Default: 300 (print quality).

    Returns:
        Path to the saved figure file.
    """
    logger.info("Creating nested CV diagram...")

    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 6)
    ax.set_ylim(1.4, 3.5)
    ax.axis("off")

    # Colors
    outer_color = "#4472C4"  # Blue - outer container
    train_outer_color = "#70AD47"  # Green - outer training section
    inner_train_color = "#ED7D31"  # Orange - inner training folds
    val_color = "#FFC000"    # Gold/Yellow - validation fold
    test_color = "#D3D3D3"   # Gray - test set

    # ===== OUTER FOLD =====
    # Blue container
    outer_box = FancyBboxPatch(
        (0.3, 2.0), 5.4, 1.2,
        boxstyle="round,pad=0.03",
        facecolor=outer_color,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(outer_box)

    # Title above
    ax.text(3, 3.35, "Outer Fold (1 of 10)", ha="center", va="center",
            fontsize=12, fontweight="bold")

    # Training section (green, left side)
    train_box = Rectangle((0.5, 2.15), 3.2, 0.9,
                           facecolor=train_outer_color, edgecolor="black", linewidth=1.5)
    ax.add_patch(train_box)
    ax.text(2.1, 2.9, "Training (80%)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    # Inner folds inside training (3 small boxes) - use orange for train, yellow for val
    fold_colors = [val_color, inner_train_color, inner_train_color]  # First is validation
    fold_labels = ["Val", "Train", "Train"]
    for i, (c, lbl) in enumerate(zip(fold_colors, fold_labels)):
        x = 0.7 + i * 1.0
        box = Rectangle((x, 2.25), 0.8, 0.45,
                         facecolor=c, edgecolor="black", linewidth=1)
        ax.add_patch(box)
        text_color = "black" if c == val_color else "white"
        ax.text(x + 0.4, 2.475, lbl, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)

    # Test section (gray, right side) - narrower to make arrow more visible
    test_box = Rectangle((4.3, 2.15), 1.2, 0.9,
                          facecolor=test_color, edgecolor="black", linewidth=1.5)
    ax.add_patch(test_box)
    ax.text(4.9, 2.6, "Test (20%)", ha="center", va="center",
            fontsize=11, fontweight="bold")

    # Arrow from training to test - bigger and more visible
    arrow = FancyArrowPatch((3.7, 2.6), (4.25, 2.6),
                            arrowstyle="-|>", mutation_scale=25,
                            linewidth=3, color="black")
    ax.add_patch(arrow)

    # ===== FLOW DESCRIPTION =====
    ax.text(3, 1.6, "Inner CV tunes hyperparameters → Best model evaluates on test fold",
            ha="center", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout(pad=0.1)

    # Save
    if output_path is None:
        output_path = FIGURES_DIR / "poster_nested_cv.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close()

    logger.info(f"Saved nested CV diagram to {output_path}")

    # Copy to poster figures
    poster_fig_path = Path(__file__).parent.parent.parent / "docs" / "KSSK - poster template" / "figures" / "nested_cv.png"
    poster_fig_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, poster_fig_path)
    logger.info(f"Copied nested CV diagram to {poster_fig_path}")

    return output_path


def generate_model_comparison_figure(
    output_path: Path | None = None,
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 300,
) -> Path:
    """
    Generate Figure 4: ROC-AUC comparison bar chart with error bars.

    Args:
        output_path: Path to save the figure. If None, saves to FIGURES_DIR.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch. Default: 300 (print quality).

    Returns:
        Path to the saved figure file.
    """
    logger.info("Creating model comparison figure (ROC-AUC)...")

    # Load all results
    model_files = {
        "XGBoost": "XGBoost_results.csv",
        "Random Forest": "RandomForest_results.csv",
        "Logistic Regression": "LogisticRegression_results.csv",
        "Naive Bayes": "NaiveBayes_results.csv",
        "k-NN": "KNN_results.csv",
    }

    results = []
    for model_name, filename in model_files.items():
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            results.append({
                "model": model_name,
                "mean_auc": df["roc_auc"].mean(),
                "std_auc": df["roc_auc"].std(),
            })
        else:
            logger.warning(f"Results file not found: {filepath}")

    if not results:
        raise FileNotFoundError("No results files found in RESULTS_DIR")

    # Create DataFrame and sort by AUC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_auc", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color scheme
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#7030A0", "#FFC000"]
    if len(results_df) > len(colors):
        colors = colors * (len(results_df) // len(colors) + 1)

    # Create horizontal bar chart
    y_pos = range(len(results_df))
    bars = ax.barh(
        y_pos,
        results_df["mean_auc"],
        xerr=results_df["std_auc"],
        color=colors[:len(results_df)],
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        error_kw={"linewidth": 2},
    )

    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(
        zip(bars, results_df["mean_auc"], results_df["std_auc"], strict=True)
    ):
        width = bar.get_width()
        ax.text(
            width + std_val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.3f} ± {std_val:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["model"], fontsize=11, fontweight="bold")
    ax.set_xlabel("ROC-AUC", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0.8, 1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = FIGURES_DIR / "poster_model_comparison.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved model comparison figure to {output_path}")

    # Copy to poster figures
    poster_fig_path = Path(__file__).parent.parent.parent / "docs" / "KSSK - poster template" / "figures" / "model_comparison.png"
    poster_fig_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, poster_fig_path)
    logger.info(f"Copied model comparison figure to {poster_fig_path}")

    return output_path


def generate_metrics_comparison_figure(
    output_path: Path | None = None,
    figsize: tuple[float, float] = (7, 5),
    dpi: int = 300,
) -> Path:
    """
    Generate Figure 5: Multi-metric comparison (Precision, Recall, F1, AUC).

    Args:
        output_path: Path to save the figure. If None, saves to FIGURES_DIR.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch. Default: 300 (print quality).

    Returns:
        Path to the saved figure file.
    """
    logger.info("Creating metrics comparison figure...")

    # Load all results
    model_files = {
        "XGBoost": "XGBoost_results.csv",
        "Random Forest": "RandomForest_results.csv",
        "Logistic Regression": "LogisticRegression_results.csv",
        "Naive Bayes": "NaiveBayes_results.csv",
        "k-NN": "KNN_results.csv",
    }

    metrics_data = []
    for model_name, filename in model_files.items():
        filepath = RESULTS_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            metrics_data.append({
                "model": model_name,
                "precision_mean": df["precision"].mean(),
                "precision_std": df["precision"].std(),
                "recall_mean": df["recall"].mean(),
                "recall_std": df["recall"].std(),
                "f1_mean": df["f1"].mean(),
                "f1_std": df["f1"].std(),
                "auc_mean": df["roc_auc"].mean(),
                "auc_std": df["roc_auc"].std(),
            })
        else:
            logger.warning(f"Results file not found: {filepath}")

    if not metrics_data:
        raise FileNotFoundError("No results files found in RESULTS_DIR")

    # Create DataFrame and sort by AUC
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values("auc_mean", ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Set up grouped bar chart
    x = np.arange(len(metrics_df))
    width = 0.2
    metrics = ["Precision", "Recall", "F1", "AUC"]
    metric_cols = ["precision_mean", "recall_mean", "f1_mean", "auc_mean"]
    metric_stds = ["precision_std", "recall_std", "f1_std", "auc_std"]

    colors = ["#4472C4", "#70AD47", "#ED7D31", "#7030A0"]

    for i, (metric, col, std_col, color) in enumerate(
        zip(metrics, metric_cols, metric_stds, colors, strict=True)
    ):
        means = metrics_df[col].values
        stds = metrics_df[std_col].values
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=metric,
            color=color,
            edgecolor="black",
            linewidth=1,
            capsize=3,
            error_kw={"linewidth": 1.5},
        )

    # Styling
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Performance Metrics Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["model"], fontsize=10, rotation=15, ha="right")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        output_path = FIGURES_DIR / "poster_metrics_comparison.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved metrics comparison figure to {output_path}")

    # Copy to poster figures
    poster_fig_path = Path(__file__).parent.parent.parent / "docs" / "KSSK - poster template" / "figures" / "roc_curves.png"
    poster_fig_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, poster_fig_path)
    logger.info(f"Copied metrics comparison figure to {poster_fig_path}")

    return output_path


def generate_wilcoxon_heatmap(
    output_path: Path | None = None,
    alpha: float = 0.05,
    figsize: tuple[float, float] = (6, 4.8),
    dpi: int = 300,
) -> Path:
    """
    Generate a poster-ready heatmap of pairwise Wilcoxon signed-rank tests (p-values).

    Uses ROC-AUC scores across outer CV folds (paired by fold index).
    """
    logger.info("Creating Wilcoxon heatmap (pairwise ROC-AUC comparisons)...")

    model_files = {
        "XGBoost": "XGBoost_results.csv",
        "Random Forest": "RandomForest_results.csv",
        "Logistic Regression": "LogisticRegression_results.csv",
        "Naive Bayes": "NaiveBayes_results.csv",
        "k-NN": "KNN_results.csv",
    }

    model_names: list[str] = []
    auc_scores: list[np.ndarray] = []

    for model_name, filename in model_files.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        df = pd.read_csv(filepath)
        if "roc_auc" not in df.columns:
            raise ValueError(f"Missing 'roc_auc' column in {filepath}")

        model_names.append(model_name)
        auc_scores.append(df["roc_auc"].to_numpy())

    n = len(model_names)
    pvals = np.full((n, n), np.nan, dtype=float)
    sig = np.full((n, n), np.nan, dtype=float)  # 1.0=significant, 0.0=not

    for i in range(n):
        for j in range(n):
            if i <= j:
                continue  # lower triangle only
            s1 = auc_scores[i]
            s2 = auc_scores[j]
            if len(s1) != len(s2):
                raise ValueError(
                    f"Fold count mismatch for {model_names[i]} vs {model_names[j]}: "
                    f"{len(s1)} vs {len(s2)}"
                )

            try:
                _stat, p = stats.wilcoxon(s1, s2)
            except ValueError:
                # e.g., all differences are zero; treat as not significant
                p = 1.0

            pvals[i, j] = float(p)
            sig[i, j] = 1.0 if p < alpha else 0.0

    # Plot: color encodes significance; annotations show p-values
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = ListedColormap(["#E6E6E6", "#70AD47"])  # gray, green
    sig_masked = np.ma.masked_invalid(sig)
    im = ax.imshow(sig_masked, cmap=cmap, vmin=0, vmax=1)

    ax.set_title("Wilcoxon Signed-Rank Test (ROC-AUC)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(model_names, fontsize=10, fontweight="bold")

    # Grid lines for readability
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    def _format_p(p: float) -> str:
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    # Annotate p-values (lower triangle)
    for i in range(n):
        for j in range(n):
            if i <= j:
                continue
            p = pvals[i, j]
            if np.isnan(p):
                continue
            is_sig = p < alpha
            ax.text(
                j,
                i,
                _format_p(p),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if is_sig else "black",
            )

    # Legend (significant vs not)
    legend_handles = [
        mpatches.Patch(facecolor="#70AD47", edgecolor="black", label=f"p < {alpha:.2f} (significant)"),
        mpatches.Patch(facecolor="#E6E6E6", edgecolor="black", label=f"p ≥ {alpha:.2f} (not significant)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()

    if output_path is None:
        output_path = FIGURES_DIR / "wilcoxon_heatmap.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(f"Saved Wilcoxon heatmap to {output_path}")

    # Copy to poster figures
    poster_fig_path = (
        Path(__file__).parent.parent.parent
        / "docs"
        / "KSSK - poster template"
        / "figures"
        / "wilcoxon_heatmap.png"
    )
    poster_fig_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, poster_fig_path)
    logger.info(f"Copied Wilcoxon heatmap to {poster_fig_path}")

    return output_path


def main():
    """Generate all poster figures."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate poster-ready figures.")
    parser.add_argument(
        "--only",
        choices=["all", "wilcoxon"],
        default="all",
        help="Generate only a subset of figures.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Generating Poster Figures")
    logger.info("=" * 60)

    if args.only == "wilcoxon":
        logger.info("\nGenerating Wilcoxon heatmap...")
        fig_path = generate_wilcoxon_heatmap()
        logger.info(f"✓ Figure saved to: {fig_path}")
        return

    # Generate Figure 1: Class Distribution
    logger.info("\nGenerating Figure 1: Class Distribution...")
    fig_path = generate_class_distribution_figure()
    logger.info(f"✓ Figure saved to: {fig_path}")

    # Generate Figure 2: Pipeline Flowchart
    logger.info("\nGenerating Figure 2: Pipeline Flowchart...")
    fig_path2 = generate_pipeline_flowchart()
    logger.info(f"✓ Figure saved to: {fig_path2}")

    # Generate Figure 3: Nested CV Diagram
    logger.info("\nGenerating Figure 3: Nested CV Diagram...")
    fig_path3 = generate_nested_cv_diagram()
    logger.info(f"✓ Figure saved to: {fig_path3}")

    # Generate Figure 4: Model Comparison (ROC-AUC)
    logger.info("\nGenerating Figure 4: Model Comparison (ROC-AUC)...")
    fig_path4 = generate_model_comparison_figure()
    logger.info(f"✓ Figure saved to: {fig_path4}")

    # Generate Figure 5: Metrics Comparison
    logger.info("\nGenerating Figure 5: Metrics Comparison...")
    fig_path5 = generate_metrics_comparison_figure()
    logger.info(f"✓ Figure saved to: {fig_path5}")

    # Generate Figure 6: Wilcoxon heatmap
    logger.info("\nGenerating Figure 6: Wilcoxon heatmap...")
    fig_path6 = generate_wilcoxon_heatmap()
    logger.info(f"✓ Figure saved to: {fig_path6}")

    logger.info("\n" + "=" * 60)
    logger.info("All figures generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
