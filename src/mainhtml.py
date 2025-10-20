# =========================
# Insurance Claims - Full Analysis to HTML Report
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    classification_report, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import lognorm
from io import BytesIO
import base64
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ============= Funzioni utili =============
def fig_to_html():
    """Converte il grafico corrente in base64 HTML"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return f'<img src="data:image/png;base64,{base64.b64encode(buffer.read()).decode()}" width="700"><br>'

def add_title(title):
    return f"<h2 style='color:#003366;'>{title}</h2>"

def add_subtitle(title):
    return f"<h3 style='color:#00509E;'>{title}</h3>"

# ============= Inizio report =============
html = f"""
<html><head><meta charset="utf-8">
<title>Insurance Claims Analysis Report</title>
<style>
body {{font-family: Arial, sans-serif; margin: 30px; background-color:#fafafa; color:#222;}}
h1 {{color:#002244; text-align:center;}}
h2,h3 {{margin-top:25px;}}
table {{border-collapse: collapse; width: 90%; margin-bottom:20px;}}
th, td {{border: 1px solid #ccc; padding: 8px; text-align:left;}}
img {{margin-top:10px; margin-bottom:25px;}}
</style>
</head><body>
<h1>Insurance Claims - Full Report</h1>
<p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
"""

# =========================
# 1️⃣ CARICAMENTO DATI
# =========================
df = pd.read_csv("insurance_dataset.csv")
html += add_title("1️⃣ Dataset Overview")
html += f"<p>Shape: {df.shape[0]} rows × {df.shape[1]} columns</p>"
html += "<b>Columns:</b> " + ", ".join(df.columns) + "<br>"
html += df.head().to_html(index=False)

# =========================
# 2️⃣ PULIZIA BASE
# =========================
df = df.dropna(subset=["Claim_Amount"])
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
num_imputer = SimpleImputer(strategy='median')
df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Income'] = num_imputer.fit_transform(df[['Income']])
for c in ['Gender','Marital_Status','Education','Occupation']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# =========================
# 3️⃣ DISTRIBUZIONI
# =========================
html += add_title("3️⃣ Distribuzioni principali")

plt.figure()
sns.histplot(df["Claim_Amount"], bins=60, kde=True)
plt.title("Distribuzione Claim_Amount")
html += fig_to_html()

df['log_Claim_Amount'] = np.log1p(df['Claim_Amount'])
plt.figure()
sns.histplot(df["log_Claim_Amount"], bins=60, kde=True)
plt.title("Distribuzione log(1+Claim_Amount)")
html += fig_to_html()

# =========================
# 4️⃣ ANALISI CATEGORICHE
# =========================
html += add_title("4️⃣ Claim medio per categoria")
cat_cols = ['Gender', 'Marital_Status', 'Education', 'Occupation']
for c in cat_cols:
    if c in df.columns:
        agg = df.groupby(c)['Claim_Amount'].mean().sort_values(ascending=False)
        plt.figure()
        sns.barplot(x=agg.index, y=agg.values)
        plt.xticks(rotation=45)
        plt.title(f"Mean Claim Amount by {c}")
        html += add_subtitle(c)
        html += fig_to_html()

# =========================
# 5️⃣ CLUSTERING
# =========================
html += add_title("5️⃣ Segmentazione clienti (KMeans)")
features_for_clust = ['Age', 'Income', 'log_Claim_Amount']
exists_cat = [c for c in ['Gender','Marital_Status','Education','Occupation'] if c in df.columns]
df_clust = df[features_for_clust + exists_cat].copy()

for c in exists_cat:
    df_clust[c] = df_clust[c].fillna('missing')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_mat = ohe.fit_transform(df_clust[exists_cat]) if exists_cat else np.empty((len(df_clust), 0))
X_num = df_clust[features_for_clust].values
X_full = np.hstack([X_num, ohe_mat]) if ohe_mat.size else X_num
X_scaled = StandardScaler().fit_transform(X_full)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
plt.figure()
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=clusters, palette='tab10')
plt.title("Cluster visualizzati con PCA")
html += fig_to_html()

cluster_profile = df.groupby('cluster').agg({'Claim_Amount':['count','mean','median'],'Age':'mean','Income':'mean'}).round(2)
html += "<b>Cluster profile:</b><br>" + cluster_profile.to_html()

# =========================
# 6️⃣ CLASSIFICAZIONE
# =========================
html += add_title("6️⃣ Predizione High vs Low Claim")

threshold = df['Claim_Amount'].quantile(0.75)
df['high_claim'] = (df['Claim_Amount'] >= threshold).astype(int)

feature_cols = ['Age','Income','Gender','Marital_Status','Education','Occupation']
X = df[feature_cols]
y = df['high_claim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

num_features = ['Age','Income']
cat_features = [c for c in feature_cols if c not in num_features]

num_trans = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_trans = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
pre = ColumnTransformer([('num', num_trans, num_features), ('cat', cat_trans, cat_features)])
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf = Pipeline([('pre', pre), ('model', model)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_proba)
html += f"<p>ROC AUC: <b>{roc_auc:.3f}</b></p>"
html += "<pre>" + classification_report(y_test, y_pred) + "</pre>"

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve")
plt.legend()
html += fig_to_html()

# Feature importance
ohe_names = clf.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
feature_names = np.concatenate([num_features, ohe_names])
importances = clf.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
plt.figure()
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top Feature Importances")
html += fig_to_html()

# =========================
# 7️⃣ MONTE CARLO
# =========================
html += add_title("7️⃣ Simulazione Monte Carlo delle perdite aggregate")

amounts = df['Claim_Amount'].values
shape, loc, scale = lognorm.fit(amounts, floc=0)
prob_claim = (df['Claim_Amount'] > 0).mean()
N_policies, simulations = 10000, 5000
total_losses = np.zeros(simulations)
for i in range(simulations):
    has_claim = np.random.binomial(1, prob_claim, size=N_policies)
    n_claims = has_claim.sum()
    total_losses[i] = lognorm.rvs(shape, loc=0, scale=scale, size=n_claims).sum() if n_claims > 0 else 0

VaR95 = np.percentile(total_losses, 95)
ES95 = total_losses[total_losses >= VaR95].mean()
html += f"<p>VaR(95%): <b>{VaR95:,.2f}</b> | ES(95%): <b>{ES95:,.2f}</b></p>"

plt.figure()
sns.histplot(total_losses, bins=80, kde=True)
plt.title("Simulated Aggregate Losses")
html += fig_to_html()

# =========================
# 🔚 Salvataggio finale
# =========================
html += "<hr><h2>Conclusioni</h2><ul><li>Analisi esplorativa completa</li><li>Segmentazione KMeans significativa</li><li>Buon modello predittivo (RandomForest)</li><li>Monte Carlo per risk estimation</li></ul>"
html += "</body></html>"

with open("Insurance_Claims_Full_Report.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Report HTML generato: Insurance_Claims_Full_Report.html")
