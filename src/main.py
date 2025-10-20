# =========================
# Insurance Claims - Full Notebook
# - EDA, Correlation, Clustering, Classification (High vs Low claim)
# - Optional: Monte Carlo aggregate loss simulation
# =========================

# Librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    classification_report, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import lognorm
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# =========================
# 1) CARICAMENTO E PRIMA PULIZIA
# =========================
df = pd.read_csv("insurance_dataset.csv")  # sostituisci col filename se diverso
print("Shape:", df.shape)
print(df.columns.tolist())
df.head()

# Convertire nomi colonne in formato coerente (opzionale)
df.columns = [c.strip() for c in df.columns]

# Controllo valori mancanti
print("\nMissing values per colonna:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

# Tipi
print("\nTipi colonne:")
print(df.dtypes)


# =========================
# 2) CONTROLLI E TRASFORMAZIONI DI BASE
# =========================
# Assunzioni: colonne presenti: Age, Gender, Income, Marital_Status, Education, Occupation, Claim_Amount

# Drop righe con Claim_Amount mancante (se poche)
df = df[~df['Claim_Amount'].isnull()].copy()

# Correggi tipi: Age e Income in numerici
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')

# Imputazione semplice per Age/Income se poche mancanti
num_imputer = SimpleImputer(strategy='median')
df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Income'] = num_imputer.fit_transform(df[['Income']])

# Assicuriamoci che le stringhe siano stripped
for c in ['Gender', 'Marital_Status', 'Education', 'Occupation']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# Esplorazione rapida Claim_Amount
print("\nClaim_Amount summary:")
print(df['Claim_Amount'].describe())

# Visualizzare distribuzione grezza (spesso è molto asimmetrica)
plt.figure()
sns.histplot(df['Claim_Amount'], bins=60, kde=True)
plt.title("Distribuzione grezza Claim_Amount")
plt.xlabel("Claim Amount")
plt.show()

# Log-transform per gestire asimmetria (creare colonna log_claim)
df['log_Claim_Amount'] = np.log1p(df['Claim_Amount'])  # log(1+x) per evitare log(0)
plt.figure()
sns.histplot(df['log_Claim_Amount'], bins=60, kde=True)
plt.title("Distribuzione log(1 + Claim_Amount)")
plt.show()


# =========================
# 3) EDA: Mean Claim per Categoria + Boxplots
# =========================
cat_cols = ['Gender', 'Marital_Status', 'Education', 'Occupation']
num_cols = ['Age', 'Income']

# Mean claim by categorical variable
for c in cat_cols:
    if c in df.columns:
        agg = df.groupby(c)['Claim_Amount'].agg(['count','mean','median']).sort_values('mean', ascending=False)
        print(f"\n== {c} ==\n", agg.head(10))
        plt.figure()
        order = agg.index
        sns.barplot(x=agg.index, y=agg['mean'])
        plt.xticks(rotation=45)
        plt.ylabel("Mean Claim Amount")
        plt.title(f"Mean Claim Amount by {c}")
        plt.show()

# Boxplots to see variance
for c in cat_cols:
    if c in df.columns:
        plt.figure()
        sns.boxplot(x=c, y='Claim_Amount', data=df)
        plt.yscale('symlog')  # symlog se ci sono outlier estremi
        plt.title(f"Boxplot Claim_Amount by {c}")
        plt.xticks(rotation=45)
        plt.show()

# Scatter/relationship numerical
sns.scatterplot(x='Age', y='Claim_Amount', data=df, alpha=0.4)
plt.yscale('symlog')
plt.title("Claim Amount vs Age")
plt.show()

sns.scatterplot(x='Income', y='Claim_Amount', data=df, alpha=0.4)
plt.yscale('symlog')
plt.title("Claim Amount vs Income")
plt.show()


# =========================
# 4) Correlazioni
# - numeriche: Spearman (robusto per non-linearità)
# - per categoriche: tasso medio / ANOVA / Cramér's V (per assoc. tra categoriali)
# =========================
# Numeriche
corr_num = df[['Age', 'Income', 'Claim_Amount', 'log_Claim_Amount']].corr(method='spearman')
plt.figure()
sns.heatmap(corr_num, annot=True, fmt=".2f", cmap="vlag")
plt.title("Spearman correlation (numeric features)")
plt.show()

# Correlazione tra categoriche e Claim_Amount: mostrare top modalità per ciascuna
cat_summary = []
for c in cat_cols:
    if c in df.columns:
        rates = df.groupby(c)['Claim_Amount'].agg(['count','mean','median']).sort_values('mean', ascending=False)
        top = rates.head(3)
        bottom = rates.tail(3)
        cat_summary.append((c, rates))
        # print top/bottom
        print(f"\nTop categories for {c} by mean Claim_Amount:")
        print(top)
        print(f"\nBottom categories for {c}:")
        print(bottom)

# Se vuoi una misura statistica: ANOVA F-value per ogni categoriale vs Claim_Amount
from sklearn.feature_selection import f_classif
# For ANOVA we need to encode categories numerically — but here just compute one-way ANOVA via scipy
for c in cat_cols:
    if c in df.columns:
        groups = [grp['Claim_Amount'].values for name, grp in df.groupby(c)]
        fval, pval = stats.f_oneway(*groups)
        print(f"\nANOVA for {c}: F={fval:.2f}, p={pval:.3e}")


# =========================
# 5) SEGMENTAZIONE CLIENTI (Clustering)
# - Creiamo un set di features ragionevoli per clustering:
#   numeriche: Age, Income, log_Claim_Amount (o mean claim per categoria)
#   categoriali: one-hot di Education, Occupation, Gender
# - Usiamo KMeans con standardizzazione e PCA per visualizzare i cluster
# =========================
features_for_clust = ['Age', 'Income', 'log_Claim_Amount']
# Build pipeline for encoding and scaling
categorical_for_clust = ['Gender', 'Marital_Status', 'Education', 'Occupation']
exists_cat = [c for c in categorical_for_clust if c in df.columns]

# Prepare encoded dataset
df_clust = df[features_for_clust + exists_cat].copy()

# Impute any remaining missing (should be none)
for c in exists_cat:
    df_clust[c] = df_clust[c].fillna('missing')

# One-hot encode categoricals
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_mat = ohe.fit_transform(df_clust[exists_cat]) if exists_cat else np.empty((len(df_clust), 0))
ohe_cols = list(ohe.get_feature_names_out(exists_cat)) if exists_cat else []

X_num = df_clust[features_for_clust].values
X_full = np.hstack([X_num, ohe_mat]) if ohe_mat.size else X_num

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# Determine optimal K via elbow (visual)
sse = {}
K_range = range(2,8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.xlabel("k")
plt.ylabel("SSE (inertia)")
plt.title("Elbow method for KMeans")
plt.show()

# Choose k (ad es. 3) - puoi cambiare in base all'elbow
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Visualize via PCA (2D)
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)

plt.figure()
sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=clusters, palette='tab10', alpha=0.6)
plt.title(f"KMeans clusters (k={k}) projected by PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='cluster')
plt.show()

# Cluster profiling: claim mean per cluster
cluster_profile = df.groupby('cluster').agg({
    'Claim_Amount': ['count','mean','median'],
    'Age': 'mean',
    'Income': 'mean',
    'log_Claim_Amount': 'mean'
}).round(2)
print("\nCluster profiling:")
print(cluster_profile)

# Tabella con top occupations per cluster (se presente)
if 'Occupation' in df.columns:
    for cl in sorted(df['cluster'].unique()):
        top_occ = df[df['cluster']==cl]['Occupation'].value_counts().head(5)
        print(f"\nTop Occupations in cluster {cl}:\n", top_occ)


# =========================
# 6) PREDIZIONE: HIGH vs LOW CLAIM
# - Definiamo 'high claim' come quantile superiore (es. 75th percentile)
# - Costruiamo classificatore per predire se un caso è high (1) vs low(0)
# =========================
# Define threshold
q = 0.75
threshold = df['Claim_Amount'].quantile(q)
df['high_claim'] = (df['Claim_Amount'] >= threshold).astype(int)
print(f"\nThreshold for high_claim (>= {q*100:.0f}th pct): {threshold:.2f}")

# Features for model: Age, Income + categorical cols
feature_cols = ['Age', 'Income', 'Gender', 'Marital_Status', 'Education', 'Occupation']
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['high_claim']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Preprocessing pipeline: impute numerics, one-hot categoricals, scale numerics
numeric_features = ['Age','Income']
cat_features = [c for c in feature_cols if c not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, cat_features)
])

clf = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# Fit
clf.fit(X_train, y_train)

# Predict + metrics
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("\nClassification report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

# Feature importance (retrieve feature names after preprocessing)
ohe_feature_names = clf.named_steps['pre'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features) if cat_features else []
num_feature_names = numeric_features
all_feature_names = np.concatenate([num_feature_names, ohe_feature_names])
importances = clf.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)
print("\nTop feature importances:")
print(feat_imp)

plt.figure()
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top Feature Importances (Random Forest)")
plt.show()


# =========================
# 7) MONTE CARLO AGGREGATE LOSS SIMULATION
# - Usiamo la distribuzione empirica o fit (lognormal) per severity
# - Frequency: stimiamo mean frequency dai dati (e.g. sinistri per policyholder se avessi frequency)
#
# =========================

# Nota: questo blocco usa approccio basato su "per policy": stimiamo prob of claim e severity dist.
if 'Claim_Amount' in df.columns:
    # 1) Fit severity distribution (lognormal) on positive claim amounts
    amounts = df['Claim_Amount'].values
    # Force positive
    amounts = amounts[amounts > 0]
    # Fit lognormal
    shape, loc, scale = lognorm.fit(amounts, floc=0)
    print(f"\nLognormal fit params: shape={shape:.3f}, scale={scale:.2f}")

    # 2) Frequency model: estimate probability a given policy has a claim in data
    # If df is per-claim, we need number policies; if dataset is one row per policy including claim amount maybe NaN for no claim
    # Here we approximate: prob_claim = fraction of records with Claim_Amount > 0
    prob_claim = (df['Claim_Amount'] > 0).mean()
    print(f"Estimated probability of a claim per policy (observed fraction): {prob_claim:.3f}")

    # 3) Monte Carlo: simulate total loss for a portfolio of N_policies for T periods
    N_policies = 10000
    simulations = 5000
    total_losses = np.zeros(simulations, dtype=np.float64)

    for i in range(simulations):
        # decide which policies have a claim this period (Bernoulli per policy)
        has_claim = np.random.binomial(1, prob_claim, size=N_policies)
        n_claims = has_claim.sum()
        # simulate severity for each claim
        if n_claims > 0:
            sev = lognorm.rvs(shape, loc=0, scale=scale, size=n_claims)
            total_losses[i] = sev.sum()
        else:
            total_losses[i] = 0.0

    # Results
    print("\nMonte Carlo aggregate loss summary (portfolio):")
    print(pd.Series(total_losses).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
    plt.figure()
    sns.histplot(total_losses, bins=80, kde=True)
    plt.title(f"Simulated total losses for portfolio of {N_policies} policies ({simulations} iters)")
    plt.xlabel("Total loss")
    plt.show()

    VaR95 = np.percentile(total_losses, 95)
    ES95 = total_losses[total_losses >= VaR95].mean()
    print(f"VaR 95%: {VaR95:.2f}, ES 95%: {ES95:.2f}")


# =========================
# 8) CONCLUSIONI E OUTPUTS DA SALVARE
# - Puoi salvare i risultati principali e i grafici per il portfolio / report
# =========================
# Esempi di salvataggio
df.to_csv("claims_processed_with_clusters.csv", index=False)
print("\nSaved processed dataframe to claims_processed_with_clusters.csv")
# Salva cluster profile
cluster_profile.to_csv("cluster_profile.csv")
print("Saved cluster profile to cluster_profile.csv")

# Fine notebook
print("\n--- END ---")
