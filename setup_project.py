from pathlib import Path
import os
import json


def create_notebook_content(title: str, description: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": [
                    f"# {title}\n",
                    f"{description}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"language": "python"},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# TODO: Add implementation\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent

    directories = [
        "data/raw",
        "data/processed",
        "data/metadata",
        "notebooks",
        "src",
        "models",
        "app",
        "results/gradcam_samples",
    ]

    files_content = {
        ".gitignore": "venv/\ndata/raw/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n*.pth\n",
        "requirements.txt": "",
        "README.md": "# lung-cancer-detection\n\nPlaceholder description for a dual-stage lung cancer detection ML project.\n",
        "config.yaml": "",
        "src/__init__.py": "",
        "src/data_loader.py": "",
        "src/preprocessing.py": "",
        "src/cnn_model.py": "",
        "src/xgb_model.py": "",
        "src/fusion.py": "",
        "src/explainability.py": "",
        "app/streamlit_app.py": "",
    }

    notebooks = {
        "notebooks/01_data_exploration.ipynb": create_notebook_content(
            "Data Exploration", "Initial exploration of CT scan and tabular metadata."
        ),
        "notebooks/02_preprocessing.ipynb": create_notebook_content(
            "Preprocessing", "Data cleaning, normalization, and feature preparation."
        ),
        "notebooks/03_cnn_model.ipynb": create_notebook_content(
            "CNN Model", "Train and evaluate deep learning model on imaging data."
        ),
        "notebooks/04_xgboost_model.ipynb": create_notebook_content(
            "XGBoost Model", "Train and evaluate gradient boosting on engineered features."
        ),
        "notebooks/05_fusion_model.ipynb": create_notebook_content(
            "Fusion Model", "Combine CNN and XGBoost predictions for final classification."
        ),
        "notebooks/06_explainability.ipynb": create_notebook_content(
            "Explainability", "Interpret predictions using Grad-CAM and feature importance."
        ),
    }

    # Create directories.
    for directory in directories:
        dir_path = project_root / directory
        os.makedirs(dir_path, exist_ok=True)

    # Add .gitkeep files to requested empty folders.
    gitkeep_dirs = [
        "data/raw",
        "data/processed",
        "data/metadata",
        "models",
        "results/gradcam_samples",
    ]
    for directory in gitkeep_dirs:
        gitkeep_path = project_root / directory / ".gitkeep"
        gitkeep_path.touch(exist_ok=True)

    # Create standard text/python files if missing.
    for relative_path, content in files_content.items():
        file_path = project_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")

    # Create notebook files if missing.
    for relative_path, notebook_json in notebooks.items():
        notebook_path = project_root / relative_path
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        if not notebook_path.exists():
            notebook_path.write_text(
                json.dumps(notebook_json, indent=2), encoding="utf-8"
            )

    print("Project structure created successfully.")


if __name__ == "__main__":
    main()
