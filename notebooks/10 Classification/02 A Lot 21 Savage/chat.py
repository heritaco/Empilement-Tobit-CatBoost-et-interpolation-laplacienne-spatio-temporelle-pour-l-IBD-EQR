# FastSISReducer: ultra-fast pre-imputation feature screener
# pip install numpy pandas scikit-learn
from __future__ import annotations
import re, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

# ---- column inference (same rules you used) ----
def _infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], Optional[str]]:
    cols = df.columns.tolist()
    ibd_cols = [c for c in ["IBD","IBD_EQR","IBD_EQR_Status"] if c in cols]
    effort_col = "TotalAbundance_SamplingOperation" if "TotalAbundance_SamplingOperation" in cols else None
    status_cols = [c for c in cols if "Status" in c]
    chem_cols = [c for c in cols if c.startswith(("Mean90Days_","Mean180Days_","Mean1Y_")) or c.endswith("_XOMP")]
    group_cols = [c for c in ["HERlvl1Name","HERlvl1Code","CodeSite_SamplingOperations_x","CodeSite_SamplingOperations_y"] if c in cols]
    reserved = set(ibd_cols + status_cols + chem_cols + group_cols + ([effort_col] if effort_col else []))
    taxa_cols = [c for c in cols if c not in reserved and c and c[0].isalpha() and any(ch.isdigit() for ch in c)]
    if "Streamsize" in cols and "Streamsize" not in chem_cols:
        chem_cols.append("Streamsize")
    return taxa_cols, status_cols, chem_cols, group_cols, effort_col

# ---- fast helpers ----
def _spearman_on_pairs(x: pd.Series, y: pd.Series) -> float:
    # pairwise complete obs
    m = x.notna() & y.notna()
    if m.sum() < 25: 
        return 0.0
    xr = x[m].rank(method="average")
    yr = y[m].rank(method="average")
    return xr.corr(yr)

def _status_to_ord(s: pd.Series) -> pd.Series:
    return s.astype("object").map({"Bad":1,"Poor":2,"Moderate":3,"Good":4,"High":5}).fillna(0).astype(np.int16)

@dataclass
class SISConfig:
    # Hard filters
    max_missing: float = 0.98       # drop if >98% NaN
    min_n_pairs: int = 50           # need at least 50 (x,y) pairs
    min_presence_frac: float = 0.005 # drop taxa with <0.5% positives after struct zeros
    # Pooling
    pool_prefix_len: int = 3
    min_pool_size: int = 4
    min_pool_prev: float = 0.02
    # Ranking
    subsample_rows: int = 20000     # speed cap; set 0 for all rows
    top_k: int = 300                # final kept features
    corr_drop: float = 0.98         # drop redundant by |rho|>0.98 w.r.t. earlier-kept feature
    random_state: int = 42

class FastSISReducer(BaseEstimator, TransformerMixin):
    def __init__(self, cfg: SISConfig = SISConfig()):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y: pd.Series):
        taxa_cols, status_cols, chem_cols, group_cols, effort_col = _infer_columns(X)
        self.group_cols_ = group_cols
        self.effort_col_ = effort_col

        # optional subsample for speed
        if self.cfg.subsample_rows and len(X) > self.cfg.subsample_rows:
            rng = np.random.RandomState(self.cfg.random_state)
            idx = rng.choice(len(X), size=self.cfg.subsample_rows, replace=False)
            Xs, ys = X.iloc[idx], y.iloc[idx]
        else:
            Xs, ys = X, y

        # structural zeros only to compute presence
        if effort_col and effort_col in Xs.columns:
            mask_zero_eff = pd.to_numeric(Xs[effort_col], errors="coerce").fillna(0) <= 0
        else:
            mask_zero_eff = pd.Series(False, index=Xs.index)

        X0 = Xs.copy()
        if mask_zero_eff.any():
            X0.loc[mask_zero_eff, taxa_cols] = 0.0

        # 1) Hard screens
        keep_taxa = []
        for c in taxa_cols:
            s = pd.to_numeric(X0[c], errors="coerce")
            if s.isna().mean() > self.cfg.max_missing:
                continue
            pres = (s.fillna(0) > 0).mean()
            if pres < self.cfg.min_presence_frac:
                continue
            keep_taxa.append(c)

        # Pool ultra-rare taxa by prefix to salvage signal cheaply
        rare_buckets = {}
        for c in set(taxa_cols) - set(keep_taxa):
            pfx = re.sub(r'[^A-Za-z].*$', '', c)[:self.cfg.pool_prefix_len] or "RARE"
            rare_buckets.setdefault(pfx, []).append(c)

        pool_cols = []
        for pfx, cols in rare_buckets.items():
            if len(cols) < self.cfg.min_pool_size:
                continue
            colname = f"POOL_{pfx}"
            X0[colname] = pd.to_numeric(X0[cols], errors="coerce").fillna(0).sum(axis=1)
            if (X0[colname] > 0).mean() >= self.cfg.min_pool_prev:
                pool_cols.append(colname)

        # Chemistry + statuses basic screen
        chem_keep = [c for c in chem_cols if X0[c].isna().mean() <= self.cfg.max_missing]
        # 2) Rank by fast Spearman with IBD on observed pairs
        cand = keep_taxa + pool_cols + chem_keep + status_cols
        ranks = []
        for c in cand:
            if c in status_cols:
                s = _status_to_ord(X0[c])
            else:
                s = pd.to_numeric(X0[c], errors="coerce")
            n_pairs = (s.notna() & ys.notna()).sum()
            if n_pairs < self.cfg.min_n_pairs:
                continue
            r = abs(_spearman_on_pairs(s, ys))
            # weight by sqrt(n_pairs/N) to prefer stable estimates
            w = np.sqrt(n_pairs / len(X0))
            ranks.append((c, r * w, r, n_pairs))
        if not ranks:
            # fallback: keep small safe core
            self.selected_ = chem_keep[: min(50, len(chem_keep))]
            self.pools_ = {pc: [] for pc in pool_cols}
            return self

        R = pd.DataFrame(ranks, columns=["col","score","rho","n"]).sort_values("score", ascending=False)

        # 3) Redundancy pruning (very cheap): greedy |rho|>corr_drop vs already kept
        selected = []
        corr_cache = {}
        for col in R["col"]:
            if len(selected) >= self.cfg.top_k:
                break
            s_col = X0[col] if col not in status_cols else _status_to_ord(X0[col])
            s_col = pd.to_numeric(s_col, errors="coerce")
            redundant = False
            for kept in selected:
                key = (kept, col)
                if key in corr_cache:
                    rkk = corr_cache[key]
                else:
                    a = pd.to_numeric(X0[kept] if kept not in status_cols else _status_to_ord(X0[kept]), errors="coerce")
                    m = a.notna() & s_col.notna()
                    if m.sum() < 25:
                        rkk = 0.0
                    else:
                        rkk = abs(a[m].rank().corr(s_col[m].rank()))
                    corr_cache[key] = rkk
                if rkk >= self.cfg.corr_drop:
                    redundant = True
                    break
            if not redundant:
                selected.append(col)

        self.selected_ = selected
        self.status_cols_ = status_cols
        self.pools_ = {pc: rare_buckets.get(pc.replace("POOL_",""), []) for pc in pool_cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # materialize only what we kept + passthrough columns required later
        Z = X.copy()
        # structural zeros for pool building consistency
        taxa_cols, _, _, _, effort_col = _infer_columns(Z)
        if effort_col and effort_col in Z.columns:
            mask_zero_eff = pd.to_numeric(Z[effort_col], errors="coerce").fillna(0) <= 0
            if mask_zero_eff.any():
                Z.loc[mask_zero_eff, taxa_cols] = 0.0
        # build pools
        for pc, cols in self.pools_.items():
            if pc in self.selected_ and cols:
                Z[pc] = pd.to_numeric(Z[cols], errors="coerce").fillna(0).sum(axis=1)
        keep = list(dict.fromkeys(self.selected_ + self.group_cols_ + ([self.effort_col_] if self.effort_col_ else [])))
        return Z.reindex(columns=[c for c in keep if c in Z.columns])



# FeatureReducer + BestEcologicalImputer pipeline
# pip install numpy pandas scikit-learn

from __future__ import annotations
import re
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

# ---------- reuse _infer_columns and BestEcologicalImputer from earlier ----------
# If running standalone, paste your BestEcologicalImputer implementation above this line.
# For brevity here, assume it's imported:
# from best_ecological_imputer import BestEcologicalImputer, _infer_columns

# ------- Minimal copies of helpers you need if not importing ---------
def _infer_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], Optional[str]]:
    cols = df.columns.tolist()
    ibd_cols = [c for c in cols if c in {"IBD", "IBD_EQR", "IBD_EQR_Status"}]
    effort_col = "TotalAbundance_SamplingOperation" if "TotalAbundance_SamplingOperation" in cols else None
    status_cols = [c for c in cols if "Status" in c]
    chem_cols = [c for c in cols if c.startswith(("Mean90Days_","Mean180Days_","Mean1Y_")) or c.endswith("_XOMP")]
    group_cols = [c for c in ["HERlvl1Name","HERlvl1Code","CodeSite_SamplingOperations_x","CodeSite_SamplingOperations_y"] if c in cols]
    reserved = set(ibd_cols + status_cols + chem_cols + group_cols + ([effort_col] if effort_col else []))
    taxa_cols = [c for c in cols if c not in reserved and c and c[0].isalpha() and any(ch.isdigit() for ch in c)]
    if "Streamsize" in cols and "Streamsize" not in chem_cols:
        chem_cols.append("Streamsize")
    return taxa_cols, status_cols, chem_cols, group_cols, effort_col

class BestEcologicalImputer(BaseEstimator, TransformerMixin):
    # Paste your full class here in real use. Placeholder to keep this cell runnable.
    def __init__(self, taxa_cols=None, status_cols=None, chem_cols=None, group_cols=None, effort_col=None):
        self.taxa_cols = taxa_cols; self.status_cols=status_cols; self.chem_cols=chem_cols
        self.group_cols=group_cols; self.effort_col=effort_col
    def fit(self, X, y=None): self.cols_=X.columns.tolist(); return self
    def transform(self, X): return X.copy()

# ---------------- Feature reducer ----------------

@dataclass
class ReducerConfig:
    min_taxon_nonmissing_frac: float = 0.02   # keep taxa with >=2% nonmissing values
    min_taxon_presence_frac: float   = 0.01   # or with >=1% positive counts on observed entries
    pool_prefix_len: int             = 3      # pool rare taxa by first N letters (genus-like)
    min_pool_size: int               = 3      # at least 3 rare taxa to form a pool
    min_pool_prevalence: float       = 0.02   # pooled feature prevalence threshold
    max_missing_chem: float          = 0.60   # drop chem if >60% NaN before fill
    chem_corr_thresh: float          = 0.95   # Spearman |rho|>0.95 -> drop redundant
    top_k_supervised: int            = 250    # final cap on features via permutation importance
    random_state: int                = 42
    n_splits: int                    = 5      # CV splits for importance

class StructuralZerosLite:
    @staticmethod
    def apply(df: pd.DataFrame, taxa_cols: List[str], effort_col: Optional[str]) -> pd.DataFrame:
        X = df.copy()
        if effort_col and effort_col in X.columns:
            mask = pd.to_numeric(X[effort_col], errors="coerce").fillna(0) <= 0
            if mask.any():
                X.loc[mask, taxa_cols] = 0.0
        return X

class FeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, cfg: ReducerConfig = ReducerConfig()):
        self.cfg = cfg

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.y_is_nonneg_ = bool(np.all(pd.to_numeric(y, errors="coerce").fillna(0) >= 0))
        taxa_cols, status_cols, chem_cols, group_cols, effort_col = _infer_columns(X)
        self.group_cols_  = group_cols
        self.effort_col_  = effort_col

        # Stage 0: structural zeros for prevalence computation
        X0 = StructuralZerosLite.apply(X, taxa_cols, effort_col)

        # Stage 1a: unsupervised screening of taxa
        taxa_keep: List[str] = []
        rare_buckets: Dict[str, List[str]] = {}
        for c in taxa_cols:
            s = pd.to_numeric(X0[c], errors="coerce")
            nonmiss_frac = s.notna().mean()
            pres_frac = (s.dropna() > 0).mean() if nonmiss_frac > 0 else 0.0
            if (nonmiss_frac >= self.cfg.min_taxon_nonmissing_frac) or (pres_frac >= self.cfg.min_taxon_presence_frac):
                taxa_keep.append(c)
            else:
                prefix = re.sub(r'[^A-Za-z].*$', '', c)[:self.cfg.pool_prefix_len] or "RARE"
                rare_buckets.setdefault(prefix, []).append(c)

        # Stage 1b: build pooled rare features
        pools: Dict[str, List[str]] = {}
        X_pool = pd.DataFrame(index=X.index)
        for pfx, cols in rare_buckets.items():
            if len(cols) < self.cfg.min_pool_size: 
                continue
            # pooled sum ignoring NaNs
            pool_name = f"POOL_{pfx}"
            X_pool[pool_name] = pd.to_numeric(X0[cols], errors="coerce").fillna(0).sum(axis=1)
            prev = (X_pool[pool_name] > 0).mean()
            if prev >= self.cfg.min_pool_prevalence:
                pools[pool_name] = cols

        self.pools_ = pools
        pooled_cols = list(pools.keys())
        taxa_stage1 = taxa_keep + pooled_cols

        # Stage 1c: chemistry pruning by missingness then correlation
        chem1 = [c for c in chem_cols if X[c].notna().mean() >= 1 - self.cfg.max_missing_chem]
        # Spearman correlation pruning
        chem_keep = []
        dropped: Set[str] = set()
        if chem1:
            S = X[chem1].rank(method="average").corr(method="pearson", min_periods=max(10, int(0.3*len(X))))
            absS = S.abs()
            order = absS.mean().sort_values(ascending=False).index.tolist()
            for c in order:
                if c in dropped: 
                    continue
                chem_keep.append(c)
                to_drop = absS.index[(absS[c] > self.cfg.chem_corr_thresh) & (absS.index != c)].tolist()
                dropped.update(to_drop)

        # Stage 1d: statuses → ordinal ints (but keep names; conversion happens in model)
        status_keep = status_cols

        # Candidate set before supervised step
        candidate_cols = taxa_stage1 + chem_keep + status_keep
        # Ensure numeric-only for the quick model: statuses will be mapped; groups excluded here
        Xc = self._to_numeric_supervised_matrix(X0, candidate_cols, status_cols)

        # Stage 2: supervised screening with HGBT + permutation importance
        rng = np.random.RandomState(self.cfg.random_state)
        if 'CodeSite_SamplingOperations_x' in X.columns:
            groups = X['CodeSite_SamplingOperations_x'].astype(str)
            splitter = GroupKFold(n_splits=min(self.cfg.n_splits, groups.nunique()))
            cv = list(splitter.split(Xc, y, groups))
        else:
            splitter = KFold(n_splits=min(self.cfg.n_splits, 5), shuffle=True, random_state=self.cfg.random_state)
            cv = list(splitter.split(Xc, y))

        model = HistGradientBoostingRegressor(
            loss="poisson" if self.y_is_nonneg_ else "squared_error",
            max_iter=200, learning_rate=0.05, min_samples_leaf=20,
            l2_regularization=1.0, random_state=self.cfg.random_state
        )
        # Fit once on full train proxy; HGBT handles NaN
        model.fit(Xc, y)

        perm = permutation_importance(
            model, Xc, y, n_repeats=3, random_state=self.cfg.random_state, scoring=None
        )
        imp = pd.Series(perm.importances_mean, index=Xc.columns).sort_values(ascending=False)
        top = imp.head(self.cfg.top_k_supervised)
        self.selected_features_ = top.index.tolist()

        # Keep also any pool columns that were created and selected
        self.selected_taxa_ = [c for c in self.selected_features_ if c in taxa_stage1]
        self.selected_chem_ = [c for c in self.selected_features_ if c in chem_keep]
        self.selected_status_ = [c for c in self.selected_features_ if c in status_keep]

        # columns needed by the imputer even if not selected as predictors (groups/effort)
        self.required_passthrough_ = [c for c in self.group_cols_ + ([self.effort_col_] if self.effort_col_ else []) if c and c in X.columns]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        X0 = StructuralZerosLite.apply(X, self.selected_taxa_ + list(sum(self.pools_.values(), [])), self.effort_col_)
        Xo = X0.copy()
        # materialize pool columns
        for pool_name, cols in self.pools_.items():
            if pool_name in self.selected_features_:
                Xo[pool_name] = pd.to_numeric(X0[cols], errors="coerce").fillna(0).sum(axis=1)
        # return only selected + passthrough
        keep = list(dict.fromkeys(self.selected_features_ + self.required_passthrough_))
        return Xo.reindex(columns=keep)

    # ---------- utilities ----------
    @staticmethod
    def _map_status_ord(s: pd.Series) -> pd.Series:
        mapping = {"Bad":1,"Poor":2,"Moderate":3,"Good":4,"High":5}
        return s.astype("object").map(mapping).fillna(0).astype(np.int16)

    def _to_numeric_supervised_matrix(self, X: pd.DataFrame, cols: List[str], status_cols: List[str]) -> pd.DataFrame:
        Xc = pd.DataFrame(index=X.index)
        for c in cols:
            if c in status_cols:
                Xc[c] = self._map_status_ord(X[c])
            else:
                Xc[c] = pd.to_numeric(X[c], errors="coerce")
        return Xc
