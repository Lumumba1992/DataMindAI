from __future__ import annotations
from io import BytesIO
from pathlib import Path
import json
import pandas as pd

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "parquet", "dta", "sav"}

def load_dataframe(source, filename: str | None = None) -> pd.DataFrame:
    name = filename or getattr(source, "name", "upload.csv")
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if ext == "csv": return pd.read_csv(source)
    if ext in {"xlsx", "xls"}: return pd.read_excel(source)
    if ext == "json": return pd.read_json(source)
    if ext == "parquet": return pd.read_parquet(source)
    if ext == "dta": return pd.read_stata(source)
    if ext == "sav": return pd.read_spss(source)
    raise ValueError("Unable to load file")

def data_profile(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    missing = df.isna().sum().sort_values(ascending=False)
    return {
        "rows": int(len(df)), "columns": int(df.shape[1]),
        "numeric": numeric, "categorical": categorical, "datetime": datetime_cols,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": float(df.isna().mean().mean() * 100),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "missing_by_column": missing[missing > 0].to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "high_cardinality": [c for c in categorical if df[c].nunique(dropna=True) > max(20, len(df) * .5)],
        "potential_ids": [c for c in df.columns if df[c].nunique(dropna=True) >= max(10, len(df) * .95)],
    }
