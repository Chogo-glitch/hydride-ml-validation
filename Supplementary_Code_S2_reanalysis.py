"""Supplementary Code S2. Grouped re-analysis of ML-HydPARK v0.0.5 (Section 9, Table 3, Fig. 7).

Inputs (all from https://github.com/mwitman1/HEAhydrideMLv2, folder hea_h2storage/ml_data)
  --data        ML-HYDPARK_v0.0.5.csv   the 437-entry, ML-ready file (the Zenodo release,
                doi:10.5281/zenodo.10680097, holds the full 772-entry database)
  --features    train_fea_v0.0.5.csv    the 150 composition features, same row order as --data
  --mpea-ref    ML-HYDPARK_v0.0.0.csv   first release; solid-solution (SS) entries absent from it
                form the 15-entry MPEA holdout set

Reproduce the manuscript:
  python Supplementary_Code_S2_reanalysis.py --data ML-HYDPARK_v0.0.5.csv \
      --features train_fea_v0.0.5.csv --mpea-ref ML-HYDPARK_v0.0.0.csv --out reanalysis_out
(tested with Python 3.13, scikit-learn 1.9.1, numpy, pandas, scipy; about 15-20 min on two cores)

Protocol (Section 9.1)
  * Random-forest regressors (300 trees) for dH, dS and ln(Peq/P0), gradient-boosted trees as a check.
  * Splits: random 5-fold x3; 5-fold grouped by chemical system x3; leave-one-family-out
    (unlabelled entries are kept for training only); MPEA holdout.
  * Per-prediction uncertainty = spread of tree predictions, recalibrated as
    sigma_cal^2 = a*sigma^2 + b (Eq. 2), with a, b fitted by maximum likelihood to out-of-fold
    predictions inside each training set (inner folds random or grouped by chemical system).
  * T1 = |dH|/|dS| (Eq. 4) from separately predicted dH and dS, scored for labelled T1 of 200-700 K.
  * Screening window 298-353 K: precision and recall.  Coverage of 95% dH intervals, pooled and per family.

Outputs (in --out)
  predictions.csv     one row per entry, split, repeat: labels, predictions, sigma, sigma_cal, fold
  fold_metrics.csv    MAE per target, split, repeat, fold
  summary_table3.csv  Table 3 layout;  coverage_by_family.csv  Fig. 7c values;  mpea_entries.csv
"""
import argparse, re, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold

R = 8.314462618
P = argparse.ArgumentParser()
P.add_argument('--data', required=True); P.add_argument('--features')
P.add_argument('--feature-key'); P.add_argument('--out', default='reanalysis_out')
P.add_argument('--mpea-ref', default=None, help='first database release; SS entries absent from it are the MPEA set')
P.add_argument('--col-comp', default='Composition_Formula'); P.add_argument('--col-dH', default='Heat_of_Formation_kJperMolH2')
P.add_argument('--col-dS', default='Entropy_of_Formation_JperMolH2perK'); P.add_argument('--col-lnP', default='LnEquilibrium_Pressure_25C')
P.add_argument('--col-family', default='Material_Class'); P.add_argument('--col-mpea', default=None,
               help='boolean column marking MPEAs; if absent, entries with >=4 principal elements (>=5 at.%%) are used')
P.add_argument('--T-ref', type=float, default=298.15, help='temperature for ln Peq if it must be derived from dH, dS')
P.add_argument('--seed', type=int, default=0); P.add_argument('--repeats', type=int, default=3)
P.add_argument('--show-columns', action='store_true')
A = P.parse_args()

df = pd.read_csv(A.data)
def _find(cands, keys, exclude=()):
    for c in cands:
        lc = c.lower()
        if any(k in lc for k in keys) and not any(x in lc for x in exclude): return c
    return None
cols = list(df.columns)
if A.col_comp not in cols: A.col_comp = _find(cols, ('composition', 'formula', 'material', 'alloy', 'name')) or A.col_comp
if A.col_dH not in cols: A.col_dH = _find(cols, ('enthalpy', 'delta_h', 'dh')) or A.col_dH
if A.col_dS not in cols: A.col_dS = _find(cols, ('entropy', 'delta_s', 'ds')) or A.col_dS
if A.col_lnP not in cols: A.col_lnP = _find(cols, ('lnp', 'ln_p', 'ln(p', 'peq', 'pressure')) or A.col_lnP
if A.col_family not in cols: A.col_family = _find(cols, ('class', 'family', 'type', 'category')) or A.col_family
if A.col_mpea is None: A.col_mpea = _find(cols, ('mpea', 'high_entropy', 'high entropy'))
print('Columns used: composition =', A.col_comp, '| dH =', A.col_dH, '| dS =', A.col_dS,
      '| lnP =', A.col_lnP, '| family =', A.col_family, '| MPEA flag =', A.col_mpea)
if A.show_columns:
    print('data columns:', list(df.columns))
    if A.features: print('feature columns:', list(pd.read_csv(A.features).columns)[:20], '...')
    raise SystemExit

def parse_comp(s):
    toks = re.findall(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', str(s))
    d = {}
    for el, n in toks: d[el] = d.get(el, 0) + (float(n) if n else 1.0)
    tot = sum(d.values()) or 1.0
    return {k: v / tot for k, v in d.items()}

comp = df[A.col_comp].map(parse_comp)
df['chem_system'] = comp.map(lambda c: '-'.join(sorted(e for e, x in c.items() if x > 0 and e != 'H')))
df['dH'] = df[A.col_dH].abs(); df['dS'] = df[A.col_dS].abs()
if A.col_lnP in df.columns: df['lnP'] = df[A.col_lnP]
else: df['lnP'] = -df['dH'] * 1e3 / (R * A.T_ref) + df['dS'] / R
df['T1'] = df['dH'] * 1e3 / df['dS']
df['family'] = df[A.col_family].fillna('unlabelled') if A.col_family in df.columns else 'unlabelled'
if A.mpea_ref:
    ref = pd.read_csv(A.mpea_ref)
    df['mpea'] = (~df[A.col_comp].isin(ref[A.col_comp])) & (df[A.col_family] == 'SS')
elif A.col_mpea and A.col_mpea in df.columns:
    df['mpea'] = df[A.col_mpea].astype(bool)
else:
    df['mpea'] = comp.map(lambda c: sum(x >= 0.05 for e, x in c.items() if e != 'H') >= 4)
    print('WARNING: no --mpea-ref given; MPEAs defined as >=4 elements at >=5 at.% (not the manuscript set)')

if A.features:
    F = pd.read_csv(A.features)
    if A.feature_key: F = df[[A.col_comp]].merge(F, left_on=A.col_comp, right_on=A.feature_key, how='left')
    F = F.drop(columns=[c for c in F.columns if str(c).startswith('Unnamed')])
    X = F.apply(pd.to_numeric, errors='coerce').to_numpy(float)
else:
    raise SystemExit('Supply --features (the 141 published composition features).')
assert len(X) == len(df), 'feature rows must match database rows'
X = np.nan_to_num(X)
os.makedirs(A.out, exist_ok=True)
df.loc[df.mpea, [A.col_comp, A.col_family]].to_csv(f'{A.out}/mpea_entries.csv', index_label='entry')
print(f'{len(df)} entries, {X.shape[1]} features, {df.chem_system.nunique()} chemical systems, {int(df.mpea.sum())} MPEAs')
print(df.family.value_counts().to_string())

def rf(seed): return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
def tree_pred(m, X):
    allp = np.stack([t.predict(X) for t in m.estimators_]); return allp.mean(0), allp.std(0)
def fit_ab(sig, res):
    s2, r2 = sig**2, res**2
    nll = lambda p: np.sum(np.log(p[0]*s2 + p[1] + 1e-9) + r2 / (p[0]*s2 + p[1] + 1e-9))
    o = minimize(nll, [1.0, max(np.var(res) * 0.1, 1e-6)], bounds=[(1e-6, None), (0, None)], method='L-BFGS-B')
    return o.x
def recal(Xtr, ytr, gtr, seed, grouped):
    """out-of-fold tree spreads and residuals inside the training set -> a, b (Eq. 2)"""
    sig, res = np.zeros(len(ytr)), np.zeros(len(ytr))
    cv = GroupKFold(5).split(Xtr, ytr, gtr) if grouped and len(set(gtr)) >= 5 else KFold(5, shuffle=True, random_state=seed).split(Xtr)
    for a, b in cv:
        m = rf(seed).fit(Xtr[a], ytr[a]); mu, s = tree_pred(m, Xtr[b]); sig[b], res[b] = s, ytr[b] - mu
    return fit_ab(sig, res)

def splits(name, rep):
    seed = A.seed + rep; idx = np.arange(len(df))
    if name == 'random':
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(idx)): yield k, tr, te
    elif name == 'chemical_system':
        g = df.chem_system.to_numpy(); u = np.random.default_rng(seed).permutation(np.unique(g))
        gid = pd.Series(range(len(u)), index=u)[g].to_numpy()
        for k, (tr, te) in enumerate(GroupKFold(5).split(idx, groups=gid)): yield k, tr, te
    elif name == 'family':
        for fam in sorted(f for f in df.family.unique() if f != 'unlabelled'):
            te = idx[df.family.to_numpy() == fam]; yield fam, np.setdiff1d(idx, te), te
    elif name == 'mpea':
        te = idx[df.mpea.to_numpy()]; yield 'mpea', np.setdiff1d(idx, te), te

os.makedirs(A.out, exist_ok=True)
preds, fm = [], []
designs = [('random', A.repeats), ('chemical_system', A.repeats), ('family', 1), ('mpea', 1)]
for name, nrep in designs:
    for rep in range(nrep):
        for fold, tr, te in splits(name, rep):
            out = df.iloc[te][['chem_system', 'family', 'mpea', 'dH', 'dS', 'lnP', 'T1']].copy()
            out['split'], out['repeat'], out['fold'] = name, rep, fold
            for tgt in ('dH', 'dS', 'lnP'):
                y = df[tgt].to_numpy(); m = rf(A.seed + rep).fit(X[tr], y[tr]); mu, s = tree_pred(m, X[te])
                out[f'{tgt}_pred'], out[f'{tgt}_sigma'] = mu, s
                g = GradientBoostingRegressor(random_state=A.seed + rep).fit(X[tr], y[tr]); out[f'{tgt}_pred_gbt'] = g.predict(X[te])
                if tgt == 'dH':
                    for inner in ('random', 'grouped'):
                        a, b = recal(X[tr], y[tr], df.chem_system.to_numpy()[tr], A.seed + rep, inner == 'grouped')
                        out[f'dH_sigma_cal_{inner}'] = np.sqrt(a * s**2 + b)
                fm.append(dict(split=name, repeat=rep, fold=fold, target=tgt, n=len(te), mae_rf=np.mean(np.abs(mu - y[te])),
                               mae_gbt=np.mean(np.abs(g.predict(X[te]) - y[te]))))
            out['T1_pred'] = out.dH_pred * 1e3 / out.dS_pred
            preds.append(out); print(f'{name} rep {rep} fold {fold}: n={len(te)}')
pred = pd.concat(preds); pred.to_csv(f'{A.out}/predictions.csv', index_label='entry')
pd.DataFrame(fm).to_csv(f'{A.out}/fold_metrics.csv', index=False)

rows, cov = [], []
lo, hi = 298.0, 353.0
for name, _ in designs:
    p = pred[pred.split == name]; per = []
    for rep, q in p.groupby('repeat'):
        w = q[(q.T1 >= 200) & (q.T1 <= 700)]
        inside, pin = w.T1.between(lo, hi), w.T1_pred.between(lo, hi)
        per.append(dict(n=len(q), dH=np.mean(abs(q.dH_pred - q.dH)), dS=np.mean(abs(q.dS_pred - q.dS)),
                        lnP=np.mean(abs(q.lnP_pred - q.lnP)), T1=np.mean(abs(w.T1_pred - w.T1)), T1_median=np.median(abs(w.T1_pred - w.T1)),
                        dH_gbt=np.mean(abs(q.dH_pred_gbt - q.dH)), lnP_gbt=np.mean(abs(q.lnP_pred_gbt - q.lnP)),
                        precision=(inside & pin).sum() / max(pin.sum(), 1), recall=(inside & pin).sum() / max(inside.sum(), 1),
                        coverage95=np.mean(abs(q.dH_pred - q.dH) <= 1.96 * q.dH_sigma_cal_random),
                        halfwidth95=np.mean(1.96 * q.dH_sigma_cal_random)))
    d = pd.DataFrame(per); r = d.mean().to_dict(); r.update({f'{k}_sd': v for k, v in d.std().to_dict().items()})
    r['split'] = name; r['pressure_factor'] = np.exp(r['lnP']); rows.append(r)
    if name == 'family':
        for fam, q in p.groupby('family'):
            for inner in ('random', 'grouped'):
                cov.append(dict(family=fam, recalibration=inner, n=len(q),
                                coverage95=np.mean(abs(q.dH_pred - q.dH) <= 1.96 * q[f'dH_sigma_cal_{inner}'])))
pd.DataFrame(rows).to_csv(f'{A.out}/summary_table3.csv', index=False)
pd.DataFrame(cov).to_csv(f'{A.out}/coverage_by_family.csv', index=False)
print(pd.DataFrame(rows)[['split', 'n', 'dH', 'dS', 'lnP', 'pressure_factor', 'T1', 'precision', 'recall', 'coverage95']].round(2).to_string(index=False))
