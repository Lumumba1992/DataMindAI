from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def fit_ols(df, target, predictors, robust=False):
    data=df[[target]+predictors].dropna().copy(); X=sm.add_constant(pd.get_dummies(data[predictors], drop_first=True, dtype=float)); y=data[target].astype(float)
    model=sm.OLS(y,X).fit(cov_type="HC3" if robust else "nonrobust")
    table=model.summary2().tables[1].reset_index().rename(columns={"index":"term"})
    return model, table, {"n":int(len(data)),"r_squared":float(model.rsquared),"adj_r_squared":float(model.rsquared_adj),"aic":float(model.aic),"bic":float(model.bic),"durbin_watson":float(sm.stats.stattools.durbin_watson(model.resid))}

def build_preprocessor(X):
    num=X.select_dtypes(include="number").columns.tolist(); cat=X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)], remainder="drop")

def compare_regressors(df,target,predictors,test_size=.2,random_state=42,selected=None):
    data=df[[target]+predictors].dropna(); X=data[predictors]; y=data[target].astype(float); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=random_state)
    models={"Linear Regression":LinearRegression(),"Ridge":Ridge(),"Lasso":Lasso(),"Elastic Net":ElasticNet(),"Decision Tree":DecisionTreeRegressor(random_state=random_state),"Random Forest":RandomForestRegressor(n_estimators=150,random_state=random_state,n_jobs=-1),"Gradient Boosting":GradientBoostingRegressor(random_state=random_state),"Extra Trees":ExtraTreesRegressor(n_estimators=150,random_state=random_state,n_jobs=-1),"KNN":KNeighborsRegressor(),"SVR":SVR(),"HistGradientBoosting":HistGradientBoostingRegressor(random_state=random_state)}
    if selected: models={k:v for k,v in models.items() if k in selected}
    results=[]; fitted={}
    for name,est in models.items():
        pipe=Pipeline([("prep",build_preprocessor(X)),("model",est)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); results.append({"model":name,"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R2":r2_score(yte,pred),"n_test":len(yte)}); fitted[name]=(pipe,Xte,yte,pred)
    return pd.DataFrame(results).sort_values("RMSE"),fitted
