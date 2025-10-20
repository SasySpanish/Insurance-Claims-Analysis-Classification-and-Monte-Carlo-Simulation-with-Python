# `src/` – Analysis Scripts

This folder contains the Python scripts used to perform the full **Insurance Claims Analytics** workflow — from raw data cleaning to advanced modeling and risk simulation.

---

## Files Overview

### `main.py`
**Purpose:**  
Performs the complete analysis and visualization pipeline directly in Python.

**Main steps:**
1. **Data Cleaning & Preparation** – Handles missing values, ensures correct types, log-transforms claim amounts.  
2. **Exploratory Data Analysis (EDA)** – Distributions, boxplots, and mean claim by demographic groups.  
3. **Correlation Analysis** – Identifies relationships between numerical and categorical variables.  
4. **Clustering (K-Means)** – Groups similar claim records by demographic and financial features, visualized through PCA.  
5. **Classification (High vs Low Claim)** – Trains a Random Forest model to predict the probability of high-value claims.  
6. **Monte Carlo Simulation** – Uses a lognormal distribution to simulate portfolio losses and estimate Value at Risk (VaR) and Expected Shortfall (ES).

**Output:**  
- Console summaries and printed tables  
- Matplotlib/Seaborn plots displayed interactively  

Use this version if you want to **run analyses and view results directly in your IDE or Jupyter**.

---

### `mainhtml.py`
**Purpose:**  
Runs the same full workflow as `main.py`, but **automatically generates an HTML report** with all results, charts, and summary statistics embedded.

**Features:**
- Automatically converts all plots to base64 images  
- Organizes content into structured sections:
  - Dataset overview  
  - Exploratory analysis  
  - Clustering results and cluster profiling  
  - Classification performance (ROC, feature importance)  
  - Monte Carlo loss simulation  
- Generates a standalone, shareable HTML report

**Output file:**  

