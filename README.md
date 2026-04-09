# Dual-Stage Lung Cancer Detection

An ML system that detects lung cancer by combining CT scan image analysis with patient metadata risk scoring — with Grad-CAM and SHAP explainability.

---

## Status
🔧 In Development — project structure set up, implementation starting.

---

## What This Project Does
1. Analyses CT scan images using a CNN to detect suspicious nodules
2. Scores patient risk using XGBoost on metadata (age, smoking history, nodule size)
3. Combines both models using a stacking ensemble for higher accuracy
4. Explains predictions visually using Grad-CAM heatmaps and SHAP plots

---

## Tech Stack
- **Language:** Python 3.11
- **Deep Learning:** PyTorch, MONAI
- **Classical ML:** XGBoost, scikit-learn
- **Explainability:** Grad-CAM, SHAP
- **App:** Streamlit

---

## Dataset
LIDC-IDRI — 1,018 CT scans with radiologist annotations  
Download: https://www.cancerimagingarchive.net/collection/lidc-idri/

---

## Project Structure
```
lung-cancer-detection/
├── data/               # CT scans and metadata (not uploaded)
├── notebooks/          # Experiment notebooks
├── src/                # Source code
├── models/             # Saved model weights
├── app/                # Streamlit demo
└── results/            # Outputs and visualizations
```

---

## Setup
```bash
git clone https://github.com/YOURUSERNAME/lung-cancer-detection.git
cd lung-cancer-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Author
**Isumi Wickramasooriya** — [GitHub](https://github.com/isumianw) · [LinkedIn](https://www.linkedin.com/in/isumi-wickramasooriya-859a11321)
**Chamethya Yasodie** — [GitHub](https://github.com/chamethyaY) · [LinkedIn](https://www.linkedin.com/in/chamethya-yasodie-a8278a349)

---

*For research and educational purposes only. Not intended for clinical use.*
