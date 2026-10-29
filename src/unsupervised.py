from __future__ import annotations
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def numeric_matrix(df, columns): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler()).fit_transform(df[columns])
def cluster(df, columns, algorithm="K-Means", n_clusters=3, random_state=42):
    X=numeric_matrix(df,columns)
    models={"K-Means":KMeans(n_clusters=n_clusters,n_init=10,random_state=random_state),"Agglomerative":AgglomerativeClustering(n_clusters=n_clusters),"DBSCAN":DBSCAN(eps=.5,min_samples=5),"Gaussian Mixture":GaussianMixture(n_components=n_clusters,random_state=random_state)}
    model=models[algorithm]; labels=model.fit_predict(X) if algorithm != "Gaussian Mixture" else model.fit(X).predict(X)
    unique=set(labels); metrics={"clusters":len(unique),"sizes":pd.Series(labels).value_counts().sort_index().to_dict()}
    if len(unique)>1 and len(unique)<len(labels): metrics.update(silhouette=float(silhouette_score(X,labels)),davies_bouldin=float(davies_bouldin_score(X,labels)),calinski_harabasz=float(calinski_harabasz_score(X,labels)))
    coords=PCA(n_components=2,random_state=random_state).fit_transform(X); out=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"cluster":labels.astype(str)})
    return metrics,out

def pca(df,columns,n_components=2):
    X=numeric_matrix(df,columns); model=PCA(n_components=min(n_components,len(columns))).fit(X); coords=model.transform(X); return model,coords

def isolation_forest(df,columns,contamination="auto",random_state=42):
    X=numeric_matrix(df,columns); model=IsolationForest(contamination=contamination,random_state=random_state); labels=model.fit_predict(X); return pd.DataFrame({"anomaly":labels==-1,"score":model.decision_function(X)})
