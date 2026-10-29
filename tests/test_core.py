import numpy as np
import pandas as pd
from src.data_loader import data_profile
from src.eda_engine import numeric_summary, outlier_summary
from src.hypothesis_tests import run_test
from src.regression_engine import fit_ols, compare_regressors
from src.classification_engine import compare_classifiers
from src.unsupervised import cluster
from src.export_utils import csv_bytes, excel_bytes, json_bytes

def dataset():
    rng=np.random.default_rng(42); n=80
    return pd.DataFrame({"x":rng.normal(size=n),"z":rng.normal(size=n),"cat":np.where(np.arange(n)%2,"a","b"),"y":rng.normal(size=n)})

def test_profile_and_eda():
    df=dataset(); p=data_profile(df); assert p["rows"]==80; assert "x" in p["numeric"]; assert not numeric_summary(df).empty; assert "flagged" in outlier_summary(df).columns

def test_hypothesis():
    df=dataset(); r=run_test("Pearson correlation",df.x,df.z); assert "p_value" in r and 0 <= r["p_value"] <= 1

def test_regression_and_classification():
    df=dataset(); _,table,metrics=fit_ols(df,"y",["x","z"]); assert "term" in table.columns and metrics["n"]==80
    cres,_,_=compare_classifiers(df,"cat",["x","z"],selected=["Logistic Regression","Random Forest"]); assert len(cres)>=1
    rres,_=compare_regressors(df,"y",["x","z"],selected=["Linear Regression","Ridge"]); assert set(["MAE","RMSE","R2"]).issubset(rres.columns)

def test_unsupervised_and_exports():
    df=dataset(); metrics,plot=cluster(df,["x","z"],"K-Means",2); assert metrics["clusters"]==2; assert len(plot)==80
    assert len(csv_bytes(df))>0 and len(excel_bytes(df))>0 and len(json_bytes({"ok":True}))>0
