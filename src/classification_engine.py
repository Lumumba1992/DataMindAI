from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def build_preprocessor(X):
    num=X.select_dtypes(include="number").columns.tolist(); cat=X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)], remainder="drop")

def compare_classifiers(df,target,predictors,test_size=.2,random_state=42,selected=None):
    data=df[[target]+predictors].dropna(); X=data[predictors]; y=data[target]; strat=y if y.nunique()>1 and y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=strat)
    models={"Logistic Regression":LogisticRegression(max_iter=1000),"KNN":KNeighborsClassifier(),"SVM":SVC(probability=True),"Decision Tree":DecisionTreeClassifier(random_state=random_state),"Random Forest":RandomForestClassifier(n_estimators=150,random_state=random_state,n_jobs=-1,class_weight="balanced"),"LDA":LinearDiscriminantAnalysis(),"QDA":QuadraticDiscriminantAnalysis(),"AdaBoost":AdaBoostClassifier(random_state=random_state),"Extra Trees":ExtraTreesClassifier(n_estimators=150,random_state=random_state,n_jobs=-1,class_weight="balanced"),"Gradient Boosting":GradientBoostingClassifier(random_state=random_state),"Gaussian NB":GaussianNB(),"HistGradientBoosting":HistGradientBoostingClassifier(random_state=random_state)}
    if selected: models={k:v for k,v in models.items() if k in selected}
    results=[]; fitted={}; classes=y.unique().tolist()
    for name,est in models.items():
        pipe=Pipeline([("prep",build_preprocessor(X)),("model",est)])
        try:
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte) if hasattr(pipe,"predict_proba") else None
            row={"model":name,"accuracy":accuracy_score(yte,pred),"balanced_accuracy":balanced_accuracy_score(yte,pred),"precision_macro":precision_score(yte,pred,average="macro",zero_division=0),"recall_macro":recall_score(yte,pred,average="macro",zero_division=0),"f1_macro":f1_score(yte,pred,average="macro",zero_division=0),"n_test":len(yte)}
            if proba is not None and len(classes)==2:
                positive_class = getattr(pipe.named_steps["model"], "classes_", classes)[1]
                row["roc_auc"]=roc_auc_score(yte,proba[:,1]); row["pr_auc"]=average_precision_score(yte,proba[:,1], pos_label=positive_class)
            else: row["roc_auc"]=None; row["pr_auc"]=None
            results.append(row); fitted[name]=(pipe,Xte,yte,pred,proba)
        except Exception as exc: results.append({"model":name,"error":str(exc)})
    return pd.DataFrame(results).sort_values("f1_macro",ascending=False),fitted,confusion_matrix(yte, next(iter(fitted.values()))[3]) if fitted else None
