# Insurance Claims and Policy Data

This folder contains the dataset used for the project **Insurance Claims Analytics**, focused on understanding the drivers of claim amounts and assessing risk patterns among policyholders.

---

## Dataset Overview

**Source:** [Kaggle – Insurance Claims and Policy Data](https://www.kaggle.com/datasets/ravalsmit/insurance-claims-and-policy-data)  
**Records:** ~13,000  
**File:** `insurance_dataset.csv`

The dataset provides a comprehensive view of policyholders’ demographic and socioeconomic characteristics, along with the corresponding **insurance claim amounts**.  
It is primarily designed for **predictive modeling**, **risk assessment**, and **exploratory data analysis** within the insurance domain.

---

## Variable Description

| Variable | Type | Description | Example / Range |
|-----------|-------|--------------|----------------|
| `Age` | Numerical | Age of the policyholder. | Min: ~18, Max: ~79 |
| `Gender` | Categorical | Gender of the policyholder. | `Male`, `Female` |
| `Income` | Numerical | Annual income of the policyholder. | Min: ~5000 – Max: ~200000 |
| `Marital_Status` | Categorical | Marital status of the policyholder. | `Single`, `Married`|
| `Education` | Categorical | Education level of the policyholder. | `Bachelor`, `Master`, `PhD`|
| `Occupation` | Categorical | Occupation type or employment category. | `Engineer`, `CEO`, `Teacher`, `Waiter`, `Doctor`|
| `Claim_Amount` | Numerical | Amount claimed by the policyholder in currency units. | Min: 104 – Max: ~99800 |

---

## Notes

- Each record represents **a single claim** submitted by a policyholder (not full policy data).  
- The dataset focuses on **claim severity (amount claimed)** rather than frequency.  
- The mix of **numerical and categorical variables** makes it ideal for:
  - Regression and classification modeling  
  - Clustering and segmentation analysis  
  - Monte Carlo simulation of portfolio losses  

---

## Usage in the Project

The dataset is used in:
- `src/insurance_claims_full_html.py` – to perform data cleaning, EDA, clustering, classification, and risk simulation.  
- `reports/Insurance_Claims_Full_Report.html` – to visualize findings and results interactively.

---

## License

The dataset is publicly available on Kaggle for educational and research purposes.  
All rights remain with the original author: [Smit Raval](https://www.kaggle.com/ravalsmit).

