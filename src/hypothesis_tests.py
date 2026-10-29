from __future__ import annotations
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def _result(name, stat, p, n, interpretation, effect=None):
    p=float(p); p_display=f"{p:.3e}" if p > 0 else "<1e-300"
    return {"test":name,"statistic":float(stat),"p_value":p,"p_value_display":p_display,"n":int(n),"effect_size":effect,"decision":"Reject H0" if p < .05 else "Fail to reject H0","interpretation":interpretation,"limitations":"Statistical significance does not establish causality or practical importance."}

def run_test(test, x, y=None, alpha=.05):
    x=np.asarray(x.dropna() if hasattr(x,"dropna") else x, dtype=float)
    if test == "One-sample t-test": stat,p=stats.ttest_1samp(x, 0); d=float(np.mean(x)/(np.std(x,ddof=1) or np.nan)); return _result(test,stat,p,len(x),"The sample mean differs from zero at the selected alpha." if p<alpha else "Evidence is insufficient to conclude that the sample mean differs from zero.",d)
    y=np.asarray(y.dropna() if hasattr(y,"dropna") else y, dtype=float) if y is not None else None
    if test in {"Independent t-test","Welch t-test"}:
        stat,p=stats.ttest_ind(x,y,equal_var=test=="Independent t-test"); pooled=np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2)); d=float((x.mean()-y.mean())/(pooled or np.nan)); return _result(test,stat,p,len(x)+len(y),"The group means differ at the selected alpha." if p<alpha else "Evidence is insufficient to conclude that the group means differ.",d)
    if test == "Paired t-test":
        n=min(len(x),len(y)); stat,p=stats.ttest_rel(x[:n],y[:n]); d=float((x[:n]-y[:n]).mean()/((x[:n]-y[:n]).std(ddof=1) or np.nan)); return _result(test,stat,p,n,"The paired mean difference differs from zero." if p<alpha else "Evidence is insufficient to conclude that the paired mean difference differs from zero.",d)
    if test == "Mann–Whitney U": stat,p=stats.mannwhitneyu(x,y,alternative="two-sided"); return _result(test,stat,p,len(x)+len(y),"The distributions differ in location/rank at the selected alpha." if p<alpha else "Evidence is insufficient to conclude that the distributions differ.")
    if test == "Wilcoxon signed-rank": stat,p=stats.wilcoxon(x,y); return _result(test,stat,p,len(x),"The paired distributions differ in rank." if p<alpha else "Evidence is insufficient to conclude that the paired distributions differ.")
    if test == "Pearson correlation": stat,p=stats.pearsonr(x,y); return _result(test,stat,p,len(x),"There is evidence of a linear association." if p<alpha else "Evidence is insufficient to conclude a linear association.",float(stat))
    if test == "Spearman correlation": stat,p=stats.spearmanr(x,y); return _result(test,stat,p,len(x),"There is evidence of a monotonic association." if p<alpha else "Evidence is insufficient to conclude a monotonic association.",float(stat))
    if test == "Kruskal–Wallis":
        groups=[np.asarray(g,dtype=float) for g in y]; stat,p=stats.kruskal(*groups); return _result(test,stat,p,sum(map(len,groups)),"At least one group differs in rank." if p<alpha else "Evidence is insufficient to conclude a rank difference.")
    raise ValueError(f"Unsupported test: {test}")

def adjust_pvalues(p_values, method="Holm"):
    mapping={"Bonferroni":"bonferroni","Holm":"holm","Benjamini-Hochberg":"fdr_bh"}
    return multipletests(p_values, method=mapping[method])[1]
