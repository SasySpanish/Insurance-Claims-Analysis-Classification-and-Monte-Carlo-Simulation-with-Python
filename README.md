# Insurance Claims Analysis – Factors Classification and Monte Carlo Simulation

This project analyzes **insurance claims data** to study which factors (age, income, gender, education, etc.) most influence the **claim amount** and to explore **risk segmentation** through clustering, predictive modeling, and Monte Carlo simulation.

---

## Project Goal

The aim is to simulate a **real-world insurance analytics workflow**, starting from raw claim-level data and progressing through:
1. **Exploratory Data Analysis (EDA)** – uncover patterns and distributions in claim amounts.
2. **Correlation and Feature Analysis** – identify demographic and socioeconomic factors that correlate with higher losses.
3. **Clustering (Customer Segmentation)** – group similar policyholders or claims based on key characteristics.
4. **Classification (High vs Low Claim)** – predict the probability of a high-value claim.
5. **Monte Carlo Simulation** – model the potential aggregate losses in a portfolio and estimate Value at Risk (VaR) and Expected Shortfall (ES).

---

## Dataset Description

The [dataset](data/readme.md) (`insurance_dataset.csv`) contains around **13,000 claim-level observations**, each representing an **individual insurance claim**, not full policies.  
It includes demographic and socioeconomic variables along with the claim amount:

| Column | Description |
|---------|--------------|
| `Age` | Policyholder's age |
| `Gender` | Policyholder's gender |
| `Income` | Annual income |
| `Marital_Status` | Marital status |
| `Education` | Education level |
| `Occupation` | Type of occupation |
| `Claim_Amount` | Claim amount in currency units |

**Note:** Since each row represents a claim (not a policy), the analysis focuses on **severity (claim size)** rather than **frequency**.

---

## Project Workflow
[Scripts descripted also here](src/readme.md) 

### Data Cleaning and Preparation
- Handle missing values.
- Ensure consistent variable types.
- Apply log-transformation to stabilize highly skewed `Claim_Amount`.

### Exploratory Data Analysis (EDA)
- Distributions and summary statistics for `Claim_Amount` and `log_Claim_Amount`.
- Mean claim amount by categorical variables (Gender, Education, Occupation, etc.).
- Visual exploration through histograms, boxplots, and bar charts.

### Correlation and ANOVA
- Spearman correlations between numeric variables.
- ANOVA tests for categorical variables to identify features with significant impact on claim amount.

### Clustering (Customer Segmentation)
- Combine demographic and monetary features.
- Apply **K-Means clustering** with PCA visualization.
- Profile each cluster based on claim behavior, age, and income.

### Classification Model (High vs Low Claim)
- Define "High Claim" as the top 25% (`Claim_Amount ≥ 75th percentile`).
- Build a **Random Forest Classifier** to predict the likelihood of a high claim.
- Evaluate model performance using ROC-AUC, confusion matrix, and feature importance.

### Monte Carlo Simulation
- Fit a **lognormal distribution** to claim severities.
- Simulate aggregate portfolio losses for 10,000 hypothetical policies across 5,000 runs.
- Estimate **Value at Risk (VaR)** and **Expected Shortfall (ES)** at 95%.

---

## Outputs

Running the script `mainhtml.py` generates a full **[HTML report](https://sasyspanish.github.io/Insurance-Claims-Analysis-Classification-and-Monte-Carlo-Simulation-with-Python/results/Insurance_Claims_Full_Report.html)** with:
- Interactive visualizations,
- Cluster profiles,
- Feature importance plots,
- Monte Carlo simulation charts,
- Summary metrics (VaR, ES, mean losses, etc).


## Key Insights

- Claim amount distributions are **highly right-skewed**, typical of severity data.
- Certain occupations and education levels show **higher average claim values**.
- Cluster segmentation reveals **distinct customer groups** with different risk profiles.
- The Monte Carlo simulation helps estimate **aggregate portfolio risk** and tail losses.

---
