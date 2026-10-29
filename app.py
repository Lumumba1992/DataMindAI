from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from config import *
from src.data_loader import load_dataframe, data_profile
from src.eda_engine import numeric_summary,categorical_summary,correlation_matrix,missing_summary,outlier_summary,insight_summary
from src.hypothesis_tests import run_test
from src.regression_engine import fit_ols,compare_regressors
from src.classification_engine import compare_classifiers
from src.unsupervised import cluster,pca,isolation_forest
from src.ai_provider import ask_ai
from src.export_utils import csv_bytes,excel_bytes,json_bytes,pdf_bytes
load_dotenv()

st.set_page_config(page_title=APP_NAME,page_icon="📊",layout="wide")
st.markdown(f"""<style> .block-container{{padding-top:1.4rem}} .brand{{color:{NAVY};font-size:2.1rem;font-weight:800}} .subtitle{{color:{TEAL};font-size:1rem}} div[data-testid='stMetric']{{background:#f8fafc;border:1px solid #e2e8f0;padding:10px;border-radius:10px}} </style>""",unsafe_allow_html=True)
if "df" not in st.session_state: st.session_state.df=None
if "original" not in st.session_state: st.session_state.original=None
if "history" not in st.session_state: st.session_state.history=[]

def set_df(df,name): st.session_state.original=df.copy(deep=True); st.session_state.df=df.copy(deep=True); st.session_state.dataset_name=name; st.session_state.history=[]
def current(): return st.session_state.df

def sidebar():
    st.sidebar.markdown(f"<div class='brand'>{APP_NAME}</div><div class='subtitle'>Analytics workspace</div>",unsafe_allow_html=True)
    pages=["Dashboard","Upload & Manage Data","Exploratory Analysis","Hypothesis Testing","Statistical Regression","ML — Classification","ML — Regression","Unsupervised Learning","AI Data Analyst","Reports & Downloads","Settings"]
    page=st.sidebar.radio("Navigate",pages)
    if current() is not None:
        p=data_profile(current()); st.sidebar.caption(f"Dataset: {getattr(st.session_state,'dataset_name','working copy')}\n\n{p['rows']:,} rows × {p['columns']} columns")
        if st.sidebar.button("Reset working copy"): st.session_state.df=st.session_state.original.copy(deep=True); st.rerun()
    return page

def require_df():
    if current() is None: st.info("Upload a dataset or load the sample dataset to begin."); return False
    return True

def dashboard():
    st.title("DataMind AI"); st.write("A local, reproducible workspace for exploratory statistics, modeling, and explainable analysis.")
    if not require_df(): return
    p=data_profile(current()); cols=st.columns(5)
    for c,label,val in zip(cols,["Rows","Columns","Missing cells","Duplicates","Memory"],[p['rows'],p['columns'],p['missing_cells'],p['duplicates'],f"{p['memory_mb']:.1f} MB"]): c.metric(label,val)
    st.subheader("Analysis readiness"); st.write("The working dataset is preserved separately from the immutable original. No rows are removed automatically.")
    st.dataframe(current().head(10),use_container_width=True)
    st.subheader("Automated quality signals"); [st.write("• "+x) for x in insight_summary(current())]

def upload():
    st.title("Upload & Manage Data")
    f=st.file_uploader("Upload CSV, Excel, JSON, Parquet, Stata, or SPSS",type=["csv","xlsx","xls","json","parquet","dta","sav"])
    c1,c2=st.columns(2)
    if c1.button("Load sample dataset"):
        from sklearn.datasets import load_iris
        d=load_iris(as_frame=True); set_df(d.frame,"iris_sample.csv"); st.success("Loaded the Iris sample dataset.")
    if f:
        try: set_df(load_dataframe(f),f.name); st.success(f"Loaded {f.name} into an isolated working copy.")
        except Exception as e: st.error(str(e))
    if not require_df(): return
    df=current(); st.write(f"**{getattr(st.session_state,'dataset_name','dataset')}** — {len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20),use_container_width=True)
    with st.expander("Working-copy transformations"):
        drop=st.multiselect("Drop columns",df.columns)
        if st.button("Apply selected column drops"):
            st.session_state.df=df.drop(columns=drop); st.session_state.history.append({"action":"drop_columns","columns":drop}); st.rerun()
        if st.button("Remove duplicate rows explicitly"):
            before=len(df); st.session_state.df=df.drop_duplicates(); st.session_state.history.append({"action":"drop_duplicates","removed":before-len(st.session_state.df)}); st.rerun()
    st.download_button("Download working dataset",current().to_csv(index=False),"datamind_cleaned.csv","text/csv")

def eda():
    st.title("Exploratory Analysis");
    if not require_df(): return
    df=current(); tab1,tab2,tab3,tab4=st.tabs(["Overview","Distributions","Relationships","Outliers"])
    with tab1:
        st.dataframe(pd.DataFrame({"metric":["Rows","Columns","Missing cells","Duplicate rows","Memory MB"],"value":[len(df),df.shape[1],df.isna().sum().sum(),df.duplicated().sum(),round(df.memory_usage(deep=True).sum()/1024**2,2)]}),hide_index=True)
        st.subheader("Numeric summary"); ns=numeric_summary(df); st.dataframe(ns,use_container_width=True); st.download_button("Download numeric summary",csv_bytes(ns),"numeric_summary.csv")
        cs=categorical_summary(df)
        if not cs.empty: st.subheader("Categorical summary"); st.dataframe(cs,use_container_width=True)
        ms=missing_summary(df)
        if not ms.empty: st.subheader("Missingness"); st.dataframe(ms,use_container_width=True); st.plotly_chart(px.bar(ms.reset_index(),x="index",y="missing",title="Missing values by column"),use_container_width=True)
        st.subheader("Evidence-based insights"); [st.write("• "+x) for x in insight_summary(df)]
    with tab2:
        nums=df.select_dtypes(include="number").columns.tolist(); cats=df.select_dtypes(exclude="number").columns.tolist()
        if nums:
            col=st.selectbox("Numeric variable",nums); st.plotly_chart(px.histogram(df,x=col,marginal="box",title=f"Distribution of {col}"),use_container_width=True)
        if cats:
            col=st.selectbox("Categorical variable",cats); st.plotly_chart(px.histogram(df,x=col,title=f"Counts of {col}"),use_container_width=True)
    with tab3:
        nums=df.select_dtypes(include="number").columns.tolist()
        if len(nums)>=2:
            corr_method=st.selectbox("Correlation method",["pearson","spearman","kendall"]); corr=correlation_matrix(df,corr_method); st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title=f"{corr_method.title()} correlation"),use_container_width=True)
            x,y=st.selectbox("X",nums),st.selectbox("Y",nums,index=1); st.plotly_chart(px.scatter(df,x=x,y=y,title=f"{x} vs {y}"),use_container_width=True)
        else: st.info("At least two numeric columns are required.")
    with tab4:
        method=st.selectbox("Method",["IQR","Z-score"]); out=outlier_summary(df,method); st.dataframe(out,use_container_width=True)

def hypothesis():
    st.title("Hypothesis Testing")
    if not require_df(): return
    df=current(); nums=df.select_dtypes(include="number").columns.tolist(); test=st.selectbox("Test",["One-sample t-test","Independent t-test","Welch t-test","Paired t-test","Mann–Whitney U","Wilcoxon signed-rank","Pearson correlation","Spearman correlation"]); xcol=st.selectbox("Variable 1",nums); ycol=st.selectbox("Variable 2 / paired variable",nums,index=min(1,len(nums)-1)) if len(nums)>1 else None
    if st.button("Run test",type="primary"):
        try:
            r=run_test(test,df[xcol],df[ycol] if test != "One-sample t-test" else None); st.json(r); st.session_state.history.append({"analysis":"hypothesis","result":r})
        except Exception as e: st.error(f"Test could not be run: {e}")

def stat_reg():
    st.title("Statistical Regression")
    if not require_df(): return
    df=current(); nums=df.select_dtypes(include="number").columns.tolist(); target=st.selectbox("Outcome",nums); preds=st.multiselect("Predictors",[c for c in nums if c!=target],default=[c for c in nums if c!=target][:3]); robust=st.checkbox("HC3 robust standard errors")
    if st.button("Fit OLS",type="primary"):
        try:
            model,table,metrics=fit_ols(df,target,preds,robust); st.write(metrics); st.dataframe(table,use_container_width=True); st.code(model.summary().as_text()); st.download_button("Download coefficients",csv_bytes(table),"ols_coefficients.csv")
        except Exception as e: st.error(str(e))

def ml_class():
    st.title("Machine Learning — Classification")
    if not require_df(): return
    df=current(); target=st.selectbox("Target",df.columns); preds=st.multiselect("Predictors",[c for c in df.columns if c!=target],default=[c for c in df.columns if c!=target]); test=st.slider("Test size",.1,.4,.2); selected=st.multiselect("Models (empty = all)",["Logistic Regression","KNN","SVM","Decision Tree","Random Forest","LDA","QDA","AdaBoost","Extra Trees","Gradient Boosting","Gaussian NB","HistGradientBoosting"])
    if st.button("Compare classifiers",type="primary"):
        try:
            res,fitted,cm=compare_classifiers(df,target,preds,test,DEFAULT_RANDOM_STATE,selected or None); st.dataframe(res,use_container_width=True); st.download_button("Download model comparison",csv_bytes(res),"classification_results.csv")
            if cm is not None: st.plotly_chart(px.imshow(cm,text_auto=True,title="Confusion matrix (first fitted model)"),use_container_width=True)
        except Exception as e: st.error(f"Classification failed: {e}")

def ml_reg():
    st.title("Machine Learning — Regression")
    if not require_df(): return
    df=current(); nums=df.select_dtypes(include="number").columns.tolist(); target=st.selectbox("Continuous target",nums); preds=st.multiselect("Predictors",[c for c in df.columns if c!=target],default=[c for c in nums if c!=target]); selected=st.multiselect("Models (empty = all)",["Linear Regression","Ridge","Lasso","Elastic Net","Decision Tree","Random Forest","Gradient Boosting","Extra Trees","KNN","SVR","HistGradientBoosting"])
    if st.button("Compare regressors",type="primary"):
        try: res,_=compare_regressors(df,target,preds,selected=selected or None); st.dataframe(res,use_container_width=True); st.download_button("Download model comparison",csv_bytes(res),"regression_results.csv"); st.plotly_chart(px.bar(res,x="model",y="RMSE",title="Held-out test RMSE"),use_container_width=True)
        except Exception as e: st.error(f"Regression failed: {e}")

def unsupervised():
    st.title("Unsupervised Learning")
    if not require_df(): return
    df=current(); nums=df.select_dtypes(include="number").columns.tolist(); cols=st.multiselect("Numeric features",nums,default=nums[:min(4,len(nums))]); alg=st.selectbox("Clustering algorithm",["K-Means","Agglomerative","DBSCAN","Gaussian Mixture"]); k=st.slider("Clusters",2,8,3)
    if st.button("Run clustering",type="primary"):
        try:
            metrics,plot=cluster(df,cols,alg,k); st.json(metrics); st.plotly_chart(px.scatter(plot,x="PC1",y="PC2",color="cluster",title="Clusters in PCA space"),use_container_width=True)
        except Exception as e: st.error(str(e))
    st.divider(); st.subheader("Anomaly detection")
    if st.button("Run Isolation Forest"):
        try: st.dataframe(isolation_forest(df,cols),use_container_width=True)
        except Exception as e: st.error(str(e))

def ai_page():
    st.title("AI Data Analyst")
    if not require_df(): return
    df=current(); p=data_profile(df); context={"dataset":getattr(st.session_state,'dataset_name','working copy'),"profile":p,"insights":insight_summary(df)}
    for m in st.session_state.get("messages",[]): st.chat_message(m["role"]).write(m["content"])
    q=st.chat_input("Ask about the dataset, statistics, code, or model choices…")
    if q:
        st.session_state.setdefault("messages",[]).append({"role":"user","content":q}); ans=ask_ai(q,context); st.session_state.messages.append({"role":"assistant","content":ans}); st.rerun()
    if st.button("Clear conversation"): st.session_state.messages=[]; st.rerun()

def reports():
    st.title("Reports & Downloads")
    if not require_df(): return
    df=current(); p=data_profile(df); sections=[("Dataset",f"{getattr(st.session_state,'dataset_name','working copy')}: {p['rows']} rows × {p['columns']} columns."),("Insights","\n".join(insight_summary(df))),("Limitations","Automated summaries are exploratory. Validate assumptions and study design before confirmatory claims.")]
    st.download_button("Download profile JSON",json_bytes(p),"datamind_profile.json","application/json"); st.download_button("Download Excel overview",excel_bytes(pd.DataFrame([p])),"datamind_overview.xlsx"); st.download_button("Download PDF report",pdf_bytes("DataMind AI Report",sections),"datamind_report.pdf","application/pdf")

def settings():
    st.title("Settings & Configuration"); st.write("Core analytics run without an API key."); st.code("OPENAI_API_KEY=your_key\nOPENAI_API_BASE=https://api.openai.com/v1\nDATAMIND_AI_MODEL=gpt-4o-mini",language="dotenv"); st.write("Optional dependencies such as XGBoost, CatBoost, SHAP, and UMAP are intentionally not required for the default install.")

page=sidebar(); {"Dashboard":dashboard,"Upload & Manage Data":upload,"Exploratory Analysis":eda,"Hypothesis Testing":hypothesis,"Statistical Regression":stat_reg,"ML — Classification":ml_class,"ML — Regression":ml_reg,"Unsupervised Learning":unsupervised,"AI Data Analyst":ai_page,"Reports & Downloads":reports,"Settings":settings}[page]()
