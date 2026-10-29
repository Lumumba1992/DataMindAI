from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .data_loader import data_profile

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include="number").columns
    if len(cols) == 0: return pd.DataFrame()
    out = df[cols].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    out["variance"] = df[cols].var()
    out["skewness"] = df[cols].skew()
    out["excess_kurtosis"] = df[cols].kurtosis()
    out["iqr"] = out["75%"] - out["25%"]
    out["cv_pct"] = np.where(out["mean"].abs() > np.finfo(float).eps, out["std"].abs()/out["mean"].abs()*100, np.nan)
    return out.reset_index(names="variable")

def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.select_dtypes(include=["object", "category", "bool"]).columns:
        s = df[c]
        mode = s.mode(dropna=True)
        rows.append({"variable": c, "distinct": int(s.nunique(dropna=True)), "missing": int(s.isna().sum()), "mode": mode.iloc[0] if len(mode) else None, "top_share_pct": float(s.value_counts(normalize=True, dropna=True).iloc[0]*100) if s.notna().any() else np.nan})
    return pd.DataFrame(rows)

def correlation_matrix(df, method="pearson"):
    return df.select_dtypes(include="number").corr(method=method)

def missing_summary(df):
    out = pd.DataFrame({"missing": df.isna().sum(), "missing_pct": df.isna().mean()*100, "dtype": df.dtypes.astype(str)})
    return out[out.missing > 0].sort_values("missing", ascending=False)

def outlier_summary(df, method="IQR", threshold=1.5):
    rows=[]
    for c in df.select_dtypes(include="number").columns:
        s=df[c].dropna()
        if len(s)==0: continue
        if method == "Z-score":
            flags = np.abs(stats.zscore(s, nan_policy="omit")) > threshold
            low, high = s.mean()-threshold*s.std(), s.mean()+threshold*s.std()
        else:
            q1,q3=s.quantile([.25,.75]); iqr=q3-q1; low,high=q1-threshold*iqr,q3+threshold*iqr; flags=(s<low)|(s>high)
        rows.append({"variable":c,"method":method,"lower":float(low),"upper":float(high),"flagged":int(flags.sum()),"flagged_pct":float(flags.mean()*100)})
    return pd.DataFrame(rows)

def insight_summary(df):
    p=data_profile(df); insights=[]
    if p["missing_cells"]: insights.append(f"{p['missing_cells']:,} missing cells ({p['missing_pct']:.2f}% of all cells) require review; missingness alone cannot establish MCAR, MAR, or MNAR.")
    if p["duplicates"]: insights.append(f"{p['duplicates']:,} duplicate rows were detected; they have not been removed automatically.")
    if p["constant_columns"]: insights.append("Constant columns: " + ", ".join(p["constant_columns"]) + ". Consider dropping them before modeling.")
    corr=correlation_matrix(df).abs(); pairs=[]
    for i,c in enumerate(corr.columns):
        for d in corr.columns[i+1:]:
            if pd.notna(corr.loc[c,d]) and corr.loc[c,d] >= .8: pairs.append(f"{c}–{d} ({corr.loc[c,d]:.2f})")
    if pairs: insights.append("Strong numeric associations detected (association is not causation): " + ", ".join(pairs[:8]) + ".")
    if not insights: insights.append("No immediate data-quality issues were detected by the configured checks.")
    insights.append("Review assumptions, leakage risk, sample size, and practical effect magnitude before confirmatory conclusions.")
    return insights
