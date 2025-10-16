# ============================================================
# MultiRegionBoostedClassifier (accuracy-focused) + selectable models
# + persistence + auto-visuals + per-region reports
# Patched: classifier wrapper (_estimator_type, decision_function) + safe calibration
# ============================================================

from typing import Dict, Any, List, Optional, Tuple
import warnings
import os, json, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import joblib

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------- utilities ----------

def _slug(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "region"

def _normalize_model_names(models: Optional[List[str]]) -> Optional[set]:
    if not models:
        return None
    alias = {
        "HGB": "HGB",
        "HISTGRADIENTBOOSTING": "HGB",
        "LIGHTGBM": "LIGHTGBM", "LGBM": "LIGHTGBM",
        "XGBOOST": "XGBOOST", "XGB": "XGBOOST",
        "CATBOOST": "CATBOOST", "CAT": "CATBOOST",
    }
    out = set()
    for m in models:
        key = str(m).upper().replace(" ", "")
        out.add(alias.get(key, key))
    return out

def _resolve_col_case(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    low = name.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    raise KeyError(f"Column '{name}' not found (case-insensitive).")

def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="None")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols

def _feature_names_from_pre(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names: List[str] = []
    if num_cols:
        names.extend(num_cols)
    if cat_cols and "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names

def _cv_for(y: pd.Series, max_splits: int = 5) -> StratifiedKFold:
    min_class = int(y.value_counts(dropna=False).min())
    n_splits = max(2, min(max_splits, max(1, min_class)))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def _can_kfold(y: pd.Series, cv: Optional[StratifiedKFold]) -> bool:
    if cv is None:
        return False
    return int(y.value_counts().min()) >= cv.get_n_splits()

# ---------- wrapper: contiguous labels per fit + fixed-width predict_proba ----------

class ContiguousLabelWrapper(BaseEstimator, ClassifierMixin):
    """
    Remap TRAIN labels to {0..k-1}. Predictions map back to global-encoded ints.
    If global_n_classes_ is set, predict_proba returns fixed K columns in global-id order.
    """
    _estimator_type = "classifier"  # ensure sklearn treats this as a classifier

    def __init__(self, estimator: BaseEstimator, global_n_classes_: Optional[int] = None):
        self.estimator = estimator
        self.global_n_classes_ = global_n_classes_
        self.estimator_ = None
        self.classes_ = None  # global int labels present in this fit

    def get_params(self, deep=True):
        params = {"estimator": self.estimator, "global_n_classes_": self.global_n_classes_}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        est_params = {}
        for k, v in list(params.items()):
            if k == "estimator":
                self.estimator = v
            elif k == "global_n_classes_":
                self.global_n_classes_ = v
            elif k.startswith("estimator__"):
                est_params[k.split("__", 1)[1]] = v
            else:
                setattr(self, k, v)
            params.pop(k, None)
        if est_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_params)
        return self

    def fit(self, X, y, **fit_params):
        y = np.asarray(y)
        uniq = np.unique(y)                 # global-encoded ints present in TRAIN
        to_local = {g: i for i, g in enumerate(uniq)}
        y_local = np.array([to_local[val] for val in y], dtype=int)
        est = clone(self.estimator)
        est.fit(X, y_local, **fit_params)
        self.estimator_ = est
        self.classes_ = uniq
        return self

    def predict(self, X):
        y_local = self.estimator_.predict(X)
        return self.classes_[np.asarray(y_local, dtype=int)]

    def predict_proba(self, X):
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("Underlying estimator has no predict_proba.")
        P_local = self.estimator_.predict_proba(X)  # (n, k_local)
        if self.global_n_classes_ is None:
            return P_local
        n, k_glob = P_local.shape[0], int(self.global_n_classes_)
        P = np.zeros((n, k_glob), dtype=float)
        for j, g in enumerate(self.classes_):
            if 0 <= int(g) < k_glob:
                P[:, int(g)] = P_local[:, j]
        row_sums = P.sum(axis=1, keepdims=True)
        fix = (row_sums == 0).flatten()
        if np.any(fix):
            P[fix, :] = 1.0 / k_glob
        else:
            P /= np.where(row_sums == 0, 1, row_sums)
        return P

    def decision_function(self, X):
        if hasattr(self.estimator_, "decision_function"):
            return self.estimator_.decision_function(X)
        raise AttributeError("Underlying estimator has no decision_function.")

# ---------- main class (accuracy-focused boosted selection) ----------

class MultiRegionBoostedClassifier:
    """
    Per-region selection among boosted trees with label remapping and visuals:
      - HGB, LightGBM*, XGBoost*, CatBoost* (*if installed)
    Select models via constructor or per-call `models=[...]` (e.g., ["HGB","XGBoost"]).
    """

    def __init__(self, xcleans: Dict[Any, pd.DataFrame], random_state: int = 42,
                 verbose: bool = True, models: Optional[List[str]] = None):
        self.xcleans = xcleans
        self.random_state = random_state
        self.verbose = verbose
        self.results: Dict[Any, Dict[str, Any]] = {}
        self.target_column: Optional[str] = None
        self.models_allowed = _normalize_model_names(models)  # None => all available

    def _candidate_spaces(self, n_classes: Optional[int], allowed: Optional[set]) -> Dict[str, Tuple[Any, dict]]:
        rng = self.random_state
        spaces: Dict[str, Tuple[Any, dict]] = {}

        def ok(tag: str) -> bool:
            return allowed is None or tag in allowed

        # HGB
        if ok("HGB"):
            hgb = ContiguousLabelWrapper(
                HistGradientBoostingClassifier(random_state=rng),
                global_n_classes_=n_classes
            )
            hgb_grid = {
                "model__estimator__learning_rate": loguniform(1e-2, 3e-1),
                "model__estimator__max_depth": randint(2, 13),
                "model__estimator__max_leaf_nodes": randint(16, 257),
                "model__estimator__min_samples_leaf": randint(10, 201),
                "model__estimator__l2_regularization": loguniform(1e-4, 10.0),
                "model__estimator__max_bins": randint(64, 257),
            }
            spaces["HGB"] = (hgb, hgb_grid)

        # LightGBM
        if ok("LIGHTGBM"):
            try:
                from lightgbm import LGBMClassifier
                lgb = ContiguousLabelWrapper(
                    LGBMClassifier(objective="multiclass", random_state=rng, n_jobs=-1, verbose=-1),
                    global_n_classes_=n_classes
                )
                lgb_grid = {
                    "model__estimator__n_estimators": randint(300, 1201),
                    "model__estimator__learning_rate": loguniform(1e-2, 3e-1),
                    "model__estimator__num_leaves": randint(16, 257),
                    "model__estimator__max_depth": randint(-1, 16),
                    "model__estimator__min_child_samples": randint(5, 201),
                    "model__estimator__subsample": uniform(0.5, 0.5),
                    "model__estimator__colsample_bytree": uniform(0.5, 0.5),
                    "model__estimator__reg_alpha": loguniform(1e-6, 10.0),
                    "model__estimator__reg_lambda": loguniform(1e-6, 10.0),
                }
                spaces["LightGBM"] = (lgb, lgb_grid)
            except Exception:
                if self.verbose: print("[warn] LightGBM not available; skipping.")

        # XGBoost
        if ok("XGBOOST"):
            try:
                from xgboost import XGBClassifier
                xgb = ContiguousLabelWrapper(
                    XGBClassifier(objective="multi:softprob", random_state=rng, tree_method="hist",
                                  eval_metric="mlogloss", n_jobs=-1, verbosity=0),
                    global_n_classes_=n_classes
                )
                xgb_grid = {
                    "model__estimator__n_estimators": randint(300, 1201),
                    "model__estimator__learning_rate": loguniform(1e-2, 3e-1),
                    "model__estimator__max_depth": randint(3, 13),
                    "model__estimator__min_child_weight": loguniform(1e-1, 10.0),
                    "model__estimator__subsample": uniform(0.5, 0.5),
                    "model__estimator__colsample_bytree": uniform(0.5, 0.5),
                    "model__estimator__gamma": loguniform(1e-6, 10.0),
                    "model__estimator__reg_alpha": loguniform(1e-6, 10.0),
                    "model__estimator__reg_lambda": loguniform(1e-6, 10.0),
                }
                spaces["XGBoost"] = (xgb, xgb_grid)
            except Exception:
                if self.verbose: print("[warn] XGBoost not available; skipping.")

        # CatBoost
        if ok("CATBOOST"):
            try:
                from catboost import CatBoostClassifier
                cb = ContiguousLabelWrapper(
                    CatBoostClassifier(loss_function="MultiClass", random_seed=rng,
                                       verbose=False, allow_writing_files=False, thread_count=-1),
                    global_n_classes_=n_classes
                )
                cb_grid = {
                    "model__estimator__iterations": randint(300, 1201),
                    "model__estimator__learning_rate": loguniform(1e-2, 3e-1),
                    "model__estimator__depth": randint(4, 11),
                    "model__estimator__l2_leaf_reg": loguniform(1e-3, 30.0),
                    "model__estimator__bagging_temperature": loguniform(1e-2, 10.0),
                }
                spaces["CatBoost"] = (cb, cb_grid)
            except Exception:
                if self.verbose: print("[warn] CatBoost not available; skipping.")

        return spaces

    def process_missing_class(
        self,
        target_column: str = "IBD_EQR_Status",
        max_splits: int = 5,
        n_iter_per_model: int = 25,
        scoring: str = "accuracy",
        calibrate: bool = True,
        prob_method: str = "predict_proba",
        models: Optional[List[str]] = None,
    ):
        self.results.clear()
        self.target_column = target_column

        for key, Xclean in self.xcleans.items():
            try:
                y_col = _resolve_col_case(Xclean, target_column)
            except KeyError:
                if self.verbose: print(f"[skip {key}] missing target '{target_column}'"); continue
            if len(Xclean) < 5:
                if self.verbose: print(f"[skip {key}] too few rows: {len(Xclean)}"); continue

            y = Xclean[y_col]
            leak = {y_col, "IBD_EQR_Status"}
            X = Xclean.drop(columns=[c for c in leak if c in Xclean.columns], errors="ignore")

            mask_tr = y.notna()
            mask_te = ~mask_tr
            if mask_tr.sum() == 0:
                if self.verbose: print(f"[skip {key}] no labeled rows"); continue

            le = LabelEncoder()
            y_tr = y.loc[mask_tr]
            y_tr_enc = pd.Series(le.fit_transform(y_tr), index=y_tr.index)
            K = len(le.classes_)

            pre, num_cols, cat_cols = _build_preprocessor(X)
            cv = _cv_for(y_tr, max_splits=max_splits)

            allowed = _normalize_model_names(models) or self.models_allowed
            candidates = self._candidate_spaces(n_classes=K, allowed=allowed)
            if not candidates:
                raise RuntimeError("No boosted candidates available with the given 'models' setting.")

            best = {"name": None, "estimator": None, "cv_score": -np.inf, "cv_results_": None}
            for name, (estimator, grid) in candidates.items():
                pipe = Pipeline([("pre", pre), ("model", estimator)])
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=grid,
                    n_iter=n_iter_per_model,
                    scoring=scoring,
                    cv=cv,
                    refit=True,
                    n_jobs=-1,
                    verbose=0,
                    random_state=self.random_state,
                    error_score=np.nan,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X.loc[mask_tr], y_tr_enc)

                mean_score = np.nanmax(search.cv_results_["mean_test_score"])
                if self.verbose:
                    print(f"[{key}] {name}: best {scoring}={mean_score:.3f}")
                if np.isfinite(mean_score) and mean_score > best["cv_score"]:
                    best.update({"name": name, "estimator": search.best_estimator_, "cv_score": float(mean_score),
                                 "cv_results_": search.cv_results_})

            if best["estimator"] is None:
                if self.verbose: print(f"[skip {key}] no model selected")
                continue

            best_pipe: Pipeline = best["estimator"]

            # OOF diagnostics
            have_proba = False
            oof_proba = None
            if _can_kfold(y_tr, cv):
                try:
                    oof_pred_enc = cross_val_predict(best_pipe, X.loc[mask_tr], y_tr_enc, cv=cv, method="predict", n_jobs=-1)
                except Exception:
                    oof_pred_enc = cross_val_predict(best_pipe, X.loc[mask_tr], y_tr_enc, cv=cv, method="predict", n_jobs=-1)
                try:
                    oof_proba = cross_val_predict(best_pipe, X.loc[mask_tr], y_tr_enc, cv=cv, method="predict_proba", n_jobs=-1)
                    have_proba = True
                except Exception:
                    have_proba = False
                oof_pred = pd.Series(le.inverse_transform(oof_pred_enc.astype(int)), index=y_tr.index)
                oof_acc = accuracy_score(y_tr, oof_pred)
                oof_f1_macro = f1_score(y_tr, oof_pred, average="macro")
                cm = confusion_matrix(y_tr, oof_pred, labels=le.classes_)
            else:
                if self.verbose:
                    print(f"[{key}] skip OOF: min_class={int(y_tr.value_counts().min())} < folds={cv.get_n_splits()}")
                best_pipe.fit(X.loc[mask_tr], y_tr_enc)
                oof_pred_enc = best_pipe.predict(X.loc[mask_tr])
                oof_pred = pd.Series(le.inverse_transform(oof_pred_enc.astype(int)), index=y_tr.index)
                if hasattr(best_pipe, "predict_proba"):
                    try:
                        oof_proba = best_pipe.predict_proba(X.loc[mask_tr])
                        have_proba = True
                    except Exception:
                        have_proba = False
                oof_acc = accuracy_score(y_tr, oof_pred)
                oof_f1_macro = f1_score(y_tr, oof_pred, average="macro")
                cm = confusion_matrix(y_tr, oof_pred, labels=le.classes_)

            cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in le.classes_],
                                 columns=[f"pred_{c}" for c in le.classes_])

            # Refit on all labeled rows
            best_pipe.fit(X.loc[mask_tr], y_tr_enc)

            # Calibration (safe)
            calibrated_pipe = best_pipe
            if calibrate and _can_kfold(y_tr, cv):
                base = best_pipe
                final_est = base.named_steps["model"]
                if is_classifier(final_est) and (hasattr(final_est, "predict_proba") or hasattr(final_est, "decision_function")):
                    calib = CalibratedClassifierCV(final_est, cv=cv, method="isotonic")
                    calibrated_pipe = Pipeline([("pre", base.named_steps["pre"]), ("model", calib)])
                    calibrated_pipe.fit(X.loc[mask_tr], y_tr_enc)
                else:
                    if self.verbose: print(f"[{key}] skip calibration: estimator lacks proba/decision_function")
            else:
                if calibrate and self.verbose:
                    print(f"[{key}] skip calibration: min_class={int(y_tr.value_counts().min())} < folds={cv.get_n_splits()}")

            # Predict missing rows
            if mask_te.any():
                X_te = X.loc[mask_te]
                yhat_enc = calibrated_pipe.predict(X_te)
                yhat_te = pd.Series(le.inverse_transform(yhat_enc.astype(int)), index=X_te.index, name="yhat_te")

                proba_te = pd.DataFrame(index=X_te.index)
                if hasattr(calibrated_pipe, "predict_proba"):
                    P = calibrated_pipe.predict_proba(X_te)
                    inner = calibrated_pipe.named_steps["model"]
                    classes_enc = getattr(inner, "classes_", None)
                    if classes_enc is None and hasattr(inner, "base_estimator"):
                        classes_enc = getattr(inner.base_estimator, "classes_", None)
                    if classes_enc is None:
                        try:
                            classes_enc = inner.estimators_[0].classes_
                        except Exception:
                            classes_enc = np.arange(P.shape[1])
                    classes = le.inverse_transform(np.array(classes_enc, dtype=int))
                    proba_te = pd.DataFrame(P, index=X_te.index, columns=[f"proba_{c}" for c in classes])
            else:
                yhat_te = pd.Series(dtype=object, name="yhat_te")
                proba_te = pd.DataFrame()

            # Feature importances (if available)
            final_est = best_pipe.named_steps["model"]
            pre_fitted = best_pipe.named_steps["pre"]
            feat_names = _feature_names_from_pre(pre_fitted, num_cols, cat_cols)
            fi = pd.Series(dtype=float)
            try:
                est_inner = getattr(final_est, "estimator_", None)
                if est_inner is None and hasattr(final_est, "estimator"):
                    est_inner = final_est.estimator
                if est_inner is not None and hasattr(est_inner, "feature_importances_"):
                    fi = pd.Series(est_inner.feature_importances_, index=feat_names).sort_values(ascending=False)
            except Exception:
                pass

            # Filled target
            y_filled = y.copy()
            if mask_te.any():
                y_filled.loc[mask_te] = yhat_te.values

            # Store
            res = {
                "best_name": best["name"],
                "best_cv_score_acc": float(best["cv_score"]),
                "oof_acc": oof_acc,
                "oof_f1_macro": oof_f1_macro,
                "oof_confusion": cm_df,
                "classes_": le.classes_,
                "label_encoder": le,
                "model": calibrated_pipe,
                "uncalibrated_model": best_pipe,
                "n_train": int(mask_tr.sum()),
                "n_infer": int(mask_te.sum()),
                "y_tr": y_tr,
                "y_tr_enc": y_tr_enc,
                "oof_pred": pd.Series(oof_pred, index=y_tr.index, name="oof_pred"),
                "oof_proba": (oof_proba if have_proba else None),
                "proba_available": bool(have_proba),
                "yhat_te": yhat_te,
                "proba_te": proba_te,
                "y_filled": y_filled,
                "feature_importances": fi,
                "cv_results_": best.get("cv_results_", None),
            }
            # Back-compat aliases
            res["best_cv_score_bal_acc"] = res["best_cv_score_acc"]
            res["oof_bal_acc"] = res["oof_acc"]

            self.results[key] = res

            if self.verbose:
                print(f"[done {key}] best={best['name']}  OOF_acc={oof_acc:.3f}  "
                      f"OOF_f1_macro={oof_f1_macro:.3f}  infer={int(mask_te.sum())}")

    # ---- convenience ----

    def infer_table(self, key: Any) -> pd.DataFrame:
        r = self.results[key]
        idx = self.xcleans[key].index
        out = pd.DataFrame(index=idx)
        out["y_true"] = r["y_tr"]
        out["oof_pred"] = r["oof_pred"]
        out["yhat_te"] = r["yhat_te"]
        out["y_filled"] = r["y_filled"]
        out = out.join(r["proba_te"], how="left")
        return out

    def all_inferred(self) -> pd.DataFrame:
        frames = []
        for k in self.results:
            df = self.infer_table(k).reset_index().rename(columns={"index": "row_id"})
            df.insert(0, "region", k)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def filled_dataframe(self, key: Any) -> pd.DataFrame:
        if self.target_column is None:
            raise RuntimeError("Run process_missing_class first.")
        df = self.xcleans[key].copy()
        col = _resolve_col_case(df, self.target_column)
        df[col] = self.results[key]["y_filled"]
        return df

    def top_features(self, key: Any, k: int = 20) -> pd.DataFrame:
        fi = self.results[key]["feature_importances"]
        if fi.empty:
            raise ValueError("Feature importances unavailable for the selected model.")
        return fi.head(k).reset_index().rename(columns={"index": "feature", 0: "importance"})

    def print_report(self, key: Any):
        r = self.results[key]
        y_tr = r["y_tr"]
        y_pred = r["oof_pred"].reindex(y_tr.index)
        print(classification_report(y_tr, y_pred, digits=3))
        print("OOF confusion matrix:")
        print(r["oof_confusion"])

    def plot_oof_confusion(self, key: Any, savepath: Optional[str] = None, normalize: Optional[str] = None):
        cm = self.results[key]["oof_confusion"].copy()
        if normalize == "row":
            cm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
            title = f"OOF confusion (row-norm) — region {key}"
        elif normalize == "col":
            cm = cm.div(cm.sum(axis=0).replace(0, 1), axis=1)
            title = f"OOF confusion (col-norm) — region {key}"
        else:
            title = f"OOF confusion — region {key}"
        plt.figure(figsize=(5, 4))
        plt.imshow(cm.values, interpolation="nearest")
        plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
        plt.yticks(range(len(cm.index)), cm.index)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        if savepath:
            Path(savepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(savepath, dpi=150)
            plt.close()
        else:
            plt.show()

    def predict_new(self, key: Any, X_new: pd.DataFrame) -> np.ndarray:
        return self.results[key]["model"].predict(X_new)

    def predict_labels_new(self, key: Any, X_new: pd.DataFrame) -> np.ndarray:
        enc = self.results[key]["model"].predict(X_new)
        le: LabelEncoder = self.results[key]["label_encoder"]
        return le.inverse_transform(enc.astype(int))

    def predict_proba_new(self, key: Any, X_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mdl = self.results[key]["model"]
        if not hasattr(mdl, "predict_proba"):
            raise AttributeError("Model does not support predict_proba.")
        P = mdl.predict_proba(X_new)
        inner = mdl.named_steps["model"]
        classes_enc = getattr(inner, "classes_", None)
        if classes_enc is None:
            try:
                classes_enc = inner.estimators_[0].classes_
            except Exception:
                classes_enc = np.arange(P.shape[1])
        le: LabelEncoder = self.results[key]["label_encoder"]
        classes = le.inverse_transform(np.array(classes_enc, dtype=int))
        return P, classes

    # ---------- visuals helpers ----------

    @staticmethod
    def _savefig(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _export_visuals(self, key: Any, r: Dict[str, Any], viz_dir: Path, top_k_classes: int = 6, top_k_feats: int = 20):
        viz_dir.mkdir(parents=True, exist_ok=True)
        # 1) Confusion matrix (counts)
        cm = r["oof_confusion"]
        plt.figure(figsize=(6, 5))
        plt.imshow(cm.values, interpolation="nearest")
        plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
        plt.yticks(range(len(cm.index)), cm.index)
        plt.title(f"OOF Confusion — {key}")
        plt.colorbar()
        self._savefig(viz_dir / "confusion_counts.png")

        # 1b) Confusion matrix (row-normalized)
        cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm_norm.values, interpolation="nearest")
        plt.xticks(range(len(cm_norm.columns)), cm_norm.columns, rotation=45, ha="right")
        plt.yticks(range(len(cm_norm.index)), cm_norm.index)
        plt.title(f"OOF Confusion (row-normalized) — {key}")
        plt.colorbar()
        self._savefig(viz_dir / "confusion_row_norm.png")

        # 2) Classification report table -> CSV and F1 bar
        y_tr = r["y_tr"]
        y_pred = r["oof_pred"].reindex(y_tr.index)
        rep = classification_report(y_tr, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(rep).to_csv(viz_dir / "classification_report.csv", index=True)
        cls = [c for c in rep.keys() if c not in {"accuracy", "macro avg", "weighted avg"}]
        f1_vals = [rep[c]["f1-score"] for c in cls]
        plt.figure(figsize=(max(6, 0.4*len(cls)), 3.6))
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.bar(cls, f1_vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1-score")
        plt.title(f"Per-class F1 — {key}")
        self._savefig(viz_dir / "per_class_f1.png")

        # 3) ROC and PR (if probabilities available)
        if r.get("proba_available", False) and r["oof_proba"] is not None:
            P = np.asarray(r["oof_proba"])
            classes = r["classes_"]
            y_bin = label_binarize(y_tr, classes=classes)
            if P.shape[1] == len(classes) and y_bin.shape[0] == P.shape[0]:
                try:
                    auc_macro = roc_auc_score(y_bin, P, average="macro", multi_class="ovr")
                except Exception:
                    auc_macro = np.nan
                try:
                    ap_scores = [average_precision_score(y_bin[:, j], P[:, j]) for j in range(P.shape[1])]
                    ap_macro = float(np.nanmean(ap_scores))
                except Exception:
                    ap_macro = np.nan
                with open(viz_dir / "prob_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"roc_auc_macro_ovr": auc_macro, "avg_precision_macro": ap_macro}, f, indent=2)

                support = y_tr.value_counts().reindex(classes, fill_value=0)
                top_idx = np.argsort(-support.values)[:min(top_k_classes, len(classes))]

                plt.figure(figsize=(6, 5))
                for j in top_idx:
                    fpr, tpr, _ = roc_curve(y_bin[:, j], P[:, j])
                    plt.plot(fpr, tpr, label=f"ROC {classes[j]}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC OVR (top-{len(top_idx)}) — {key}")
                plt.legend(fontsize=8)
                self._savefig(viz_dir / "roc_ovr_topk.png")

                plt.figure(figsize=(6, 5))
                for j in top_idx:
                    pr, rc, _ = precision_recall_curve(y_bin[:, j], P[:, j])
                    plt.plot(rc, pr, label=f"PR {classes[j]}")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"PR OVR (top-{len(top_idx)}) — {key}")
                plt.legend(fontsize=8)
                self._savefig(viz_dir / "pr_ovr_topk.png")

                plt.figure(figsize=(6, 5))
                for j in top_idx:
                    prob_true, prob_pred = calibration_curve(y_bin[:, j], P[:, j], n_bins=10, strategy="uniform")
                    plt.plot(prob_pred, prob_true, marker="o", label=f"Cal {classes[j]}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
                plt.title(f"Reliability (top-{len(top_idx)}) — {key}")
                plt.legend(fontsize=8)
                self._savefig(viz_dir / "calibration_topk.png")

        # 4) Feature importances
        fi = r["feature_importances"]
        if isinstance(fi, pd.Series) and not fi.empty:
            k = min(top_k_feats, len(fi))
            plt.figure(figsize=(8, max(3.5, 0.25*k)))
            fi.head(k)[::-1].plot(kind="barh")
            plt.xlabel("Importance")
            plt.title(f"Top-{k} feature importances — {key}")
            self._savefig(viz_dir / "feature_importances_topk.png")

        # 5) CV score trace
        cvres = r.get("cv_results_", None)
        if cvres is not None:
            dfcv = pd.DataFrame(cvres)
            dfcv.to_csv(viz_dir / "cv_results.csv", index=False)
            if "mean_test_score" in dfcv:
                plt.figure(figsize=(6, 4))
                plt.plot(np.arange(len(dfcv["mean_test_score"])), dfcv["mean_test_score"], marker="o", linestyle="-")
                plt.xlabel("Candidate #"); plt.ylabel("mean_test_score")
                plt.title(f"RandomizedSearch scores — {key}")
                self._savefig(viz_dir / "cv_scores.png")

        # 6) Save compact metrics JSON
        metrics = {
            "oof_acc": float(r["oof_acc"]),
            "oof_f1_macro": float(r["oof_f1_macro"]),
            "n_train": int(r["n_train"]),
            "n_infer": int(r["n_infer"]),
        }
        with open(viz_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # ---------- persistence ----------

    def save_all(self, outdir: str = "models", compress: int = 3, make_figs: bool = True,
                 figs_folder_suffix: str = "_viz", top_k_classes: int = 6, top_k_feats: int = 20) -> pd.DataFrame:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        rows = []
        for key, r in self.results.items():
            best_name = (r["best_name"] or "model").lower()
            base = Path(outdir) / f"best_{best_name}_{_slug(key)}"

            joblib.dump(r["model"], f"{base}.joblib", compress=compress)
            joblib.dump(r["uncalibrated_model"], f"{base}_uncal.joblib", compress=compress)
            joblib.dump(r["label_encoder"], f"{base}_label_encoder.joblib", compress=compress)

            scores = {
                "best_cv_score_acc": r.get("best_cv_score_acc", r.get("best_cv_score_bal_acc")),
                "oof_acc": r.get("oof_acc", r.get("oof_bal_acc")),
                "oof_f1_macro": r["oof_f1_macro"],
            }
            meta = {
                "region": str(key),
                "best_name": r["best_name"],
                "paths": {
                    "calibrated": f"{base}.joblib",
                    "uncalibrated": f"{base}_uncal.joblib",
                    "label_encoder": f"{base}_label_encoder.joblib",
                    "meta": f"{base}_meta.json",
                },
                "scores": scores,
                "classes": list(map(str, r["classes_"])),
                "n_train": r["n_train"],
                "n_infer": r["n_infer"],
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "target_column": self.target_column,
            }
            with open(f"{base}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            if make_figs:
                viz_dir = Path(str(base) + figs_folder_suffix)
                try:
                    self._export_visuals(key, r, viz_dir=viz_dir, top_k_classes=top_k_classes, top_k_feats=top_k_feats)
                except Exception as e:
                    if self.verbose:
                        print(f"[warn] visuals for {key} failed: {e}")

            rows.append({
                "region": key,
                "best_name": r["best_name"],
                "calibrated_path": f"{base}.joblib",
                "encoder_path": f"{base}_label_encoder.joblib",
                "meta_path": f"{base}_meta.json",
                "viz_dir": (str(Path(str(base) + figs_folder_suffix)) if make_figs else None),
                "oof_acc": scores["oof_acc"],
                "oof_f1_macro": scores["oof_f1_macro"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def load_bundle(basepath: str):
        pipe = joblib.load(basepath + ".joblib")
        le = joblib.load(basepath + "_label_encoder.joblib")
        with open(basepath + "_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, le, meta

    # ---------- report helpers ----------

    def save_oof_report(self, key: Any, outdir: str = "models",
                        figs_folder_suffix: str = "_viz",
                        top_k_classes: int = 6, top_k_feats: int = 20) -> str:
        if key not in self.results:
            raise KeyError(f"Unknown region key: {key}")
        r = self.results[key]
        best_name = (r["best_name"] or "model").lower()
        base = Path(outdir) / f"best_{best_name}_{_slug(key)}"
        viz_dir = Path(str(base) + figs_folder_suffix)
        viz_dir.mkdir(parents=True, exist_ok=True)

        self._export_visuals(key, r, viz_dir=viz_dir,
                             top_k_classes=top_k_classes, top_k_feats=top_k_feats)

        r["oof_confusion"].to_csv(viz_dir / "confusion_matrix.csv")
        y_tr = r["y_tr"]
        y_pred = r["oof_pred"].reindex(y_tr.index)
        with open(viz_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_tr, y_pred, digits=4, zero_division=0))

        df = pd.DataFrame({"y_true": y_tr, "oof_pred": y_pred})
        if r.get("proba_available", False) and r["oof_proba"] is not None:
            P = np.asarray(r["oof_proba"])
            classes = r["classes_"]
            for j, c in enumerate(classes):
                df[f"proba_{c}"] = P[:, j]
        df.to_csv(viz_dir / "oof_predictions.csv", index=True)

        manifest = {
            "region": str(key),
            "best_name": r["best_name"],
            "report_dir": str(viz_dir),
            "n_train": int(r["n_train"]),
            "n_infer": int(r["n_infer"]),
            "oof_acc": float(r["oof_acc"]),
            "oof_f1_macro": float(r["oof_f1_macro"]),
        }
        with open(viz_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return str(viz_dir)

    def save_all_reports(self, outdir: str = "models",
                         figs_folder_suffix: str = "_viz",
                         top_k_classes: int = 6, top_k_feats: int = 20) -> pd.DataFrame:
        rows = []
        for key in self.results:
            path = self.save_oof_report(key, outdir=outdir,
                                        figs_folder_suffix=figs_folder_suffix,
                                        top_k_classes=top_k_classes, top_k_feats=top_k_feats)
            r = self.results[key]
            rows.append({
                "region": key,
                "best_name": r["best_name"],
                "report_dir": path,
                "oof_acc": float(r["oof_acc"]),
                "oof_f1_macro": float(r["oof_f1_macro"]),
            })
        return pd.DataFrame(rows)

# ----------------- usage example -----------------
# xcleans = {"region1": df1, "region2": df2, ...}
# # Skip CatBoost globally:
# auto = MultiRegionBoostedClassifier(xcleans, random_state=42, verbose=True,
#                                     models=["HGB", "LightGBM", "XGBoost"])
# auto.process_missing_class(scoring="accuracy", n_iter_per_model=25, max_splits=5, calibrate=True)
# manifest = auto.save_all(outdir="models", compress=3, make_figs=True)
# reports = auto.save_all_reports(outdir="models")
# print(manifest)
# print(reports)
