# DataMind AI

DataMind AI is a local Streamlit analytics workspace for dataset inspection, exploratory analysis, hypothesis testing, statistical regression, supervised machine learning, clustering, anomaly detection, exports, and optional dataset-aware AI assistance.

## Quick start

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\launch.bat
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

## Optional AI provider

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The application will still run all non-AI analytics without a key. Only metadata, computed summaries, and the user question are sent to the configured provider by default; raw rows are not sent automatically.

## Included workflows

- Upload CSV, Excel, JSON, Parquet, Stata, and SPSS files where the relevant pandas reader is installed.
- Preserve an immutable original dataset and operate on a separate working copy.
- Review schema, missingness, duplicates, descriptive statistics, correlations, distributions, and outlier flags.
- Run core parametric and non-parametric tests with p-values, decisions, and effect-size fields where applicable.
- Fit OLS with optional HC3 robust standard errors and inspect coefficient output.
- Compare 10+ classification models and 10+ regression models using leakage-safe preprocessing pipelines and held-out test metrics.
- Run K-Means, agglomerative clustering, DBSCAN, Gaussian mixtures, PCA-space plots, and Isolation Forest.
- Export CSV, Excel, JSON, and PDF outputs.

## Quality and limitations

This is a practical local vertical slice designed to be extensible. Optional XGBoost, CatBoost, SHAP, UMAP, repeated-measures ANOVA, advanced diagnostics, arbitrary code execution, and full report composition are not enabled in the default install. AI-generated code is displayed as text and is not executed inside the application. Statistical outputs should be reviewed against the study design and assumptions; association is not causation.

## Tests

```bash
pytest -q
```

## Troubleshooting

If a package fails to build, use Python 3.11 or 3.12 in a fresh virtual environment and upgrade pip before installing. Parquet support may require `pyarrow`; SPSS/Stata readers may require their pandas optional dependencies. For large files, increase `DATAMIND_MAX_UPLOAD_MB` and use a machine with sufficient memory.
