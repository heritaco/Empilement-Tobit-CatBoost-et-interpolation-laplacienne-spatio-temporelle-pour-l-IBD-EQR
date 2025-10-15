# ============================================================
# MultiRegionBoostedClassifier (accuracy-focused) + persistence
# ============================================================

from typing import Dict, Any, List, Optional, Tuple
import warnings
import os, json, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------- utilities ----------

def _slug(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "region"

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

# ---------- wrapper to enforce 0..k-1 labels per fit ----------

class ContiguousLabelWrapper(BaseEstimator, ClassifierMixin):
    """
    For each fit(), remap labels present in that TRAIN split to {0..k-1}.
    Predictions are mapped back to the original global integer labels.
    """

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_ = None
        self.classes_ = None  # array of global int labels present in this fit

    def get_params(self, deep=True):
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        est_params = {}
        for k, v in list(params.items()):
            if k == "estimator":
                self.estimator = v
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
        uniq = np.unique(y)
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
        return self.estimator_.predict_proba(X)

# ---------- main class (accuracy-focused boosted selection) ----------

class MultiRegionBoostedClassifier:
    """
    Per-region model selection among boosted trees with label remapping:
      - HistGradientBoostingClassifier
      - LightGBM (if installed)
      - XGBoost (if installed)
      - CatBoost (if installed)
    Selection metric defaults to accuracy.
    """

    def __init__(self, xcleans: Dict[Any, pd.DataFrame], random_state: int = 42, verbose: bool = True):
        self.xcleans = xcleans
        self.random_state = random_state
        self.verbose = verbose
        self.results: Dict[Any, Dict[str, Any]] = {}
        self.target_column: Optional[str] = None

    def _candidate_spaces(self) -> Dict[str, Tuple[Any, dict]]:
        rng = self.random_state
        spaces: Dict[str, Tuple[Any, dict]] = {}

        # HGB
        hgb = ContiguousLabelWrapper(HistGradientBoostingClassifier(random_state=rng))
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
        try:
            from lightgbm import LGBMClassifier
            lgb = ContiguousLabelWrapper(LGBMClassifier(objective="multiclass",
                                                        random_state=rng, n_jobs=-1, verbose=-1))
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
        try:
            from xgboost import XGBClassifier
            xgb = ContiguousLabelWrapper(XGBClassifier(
                objective="multi:softprob",
                random_state=rng,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1,
                verbosity=0,
            ))
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
        try:
            from catboost import CatBoostClassifier
            cb = ContiguousLabelWrapper(CatBoostClassifier(
                loss_function="MultiClass",
                random_seed=rng,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ))
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
        scoring: str = "accuracy",           # default: total accuracy
        calibrate: bool = True,
        prob_method: str = "predict_proba",
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
            leak = {y_col, "IBD", "IBD_EQR", "IBD_EQR_Status", "IBD_EQR_STATUS"}
            X = Xclean.drop(columns=[c for c in leak if c in Xclean.columns], errors="ignore")

            mask_tr = y.notna()
            mask_te = ~mask_tr
            if mask_tr.sum() == 0:
                if self.verbose: print(f"[skip {key}] no labeled rows"); continue

            # global string->int encoder
            le = LabelEncoder()
            y_tr = y.loc[mask_tr]
            y_tr_enc = pd.Series(le.fit_transform(y_tr), index=y_tr.index)

            pre, num_cols, cat_cols = _build_preprocessor(X)
            cv = _cv_for(y_tr, max_splits=max_splits)

            candidates = self._candidate_spaces()
            if not candidates:
                raise RuntimeError("No boosted candidates available.")

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
                if self.verbose: print(f"[skip {key}] no model selected"); continue

            best_pipe: Pipeline = best["estimator"]

            # OOF diagnostics
            if _can_kfold(y_tr, cv):
                try:
                    oof_pred_enc = cross_val_predict(best_pipe, X.loc[mask_tr], y_tr_enc, cv=cv, method="predict", n_jobs=-1)
                except Exception:
                    oof_pred_enc = cross_val_predict(best_pipe, X.loc[mask_tr], y_tr_enc, cv=cv, method="predict", n_jobs=-1)
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
                oof_acc = accuracy_score(y_tr, oof_pred)
                oof_f1_macro = f1_score(y_tr, oof_pred, average="macro")
                cm = confusion_matrix(y_tr, oof_pred, labels=le.classes_)

            cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in le.classes_],
                                 columns=[f"pred_{c}" for c in le.classes_])

            # Refit on all labeled rows
            best_pipe.fit(X.loc[mask_tr], y_tr_enc)

            # Calibration
            calibrated_pipe = best_pipe
            if calibrate and _can_kfold(y_tr, cv):
                base = best_pipe
                final_est = base.named_steps["model"]
                if hasattr(final_est, "predict_proba") or hasattr(final_est, "decision_function"):
                    calib = CalibratedClassifierCV(final_est, cv=cv, method="isotonic")
                    calibrated_pipe = Pipeline([("pre", base.named_steps["pre"]), ("model", calib)])
                    calibrated_pipe.fit(X.loc[mask_tr], y_tr_enc)
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
                            classes_enc = np.unique(y_tr_enc)
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
                "best_cv_score_acc": float(best["cv_score"]),   # accuracy-focused
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
                "oof_pred": pd.Series(oof_pred, index=y_tr.index, name="oof_pred"),
                "yhat_te": yhat_te,
                "proba_te": proba_te,
                "y_filled": y_filled,
                "feature_importances": fi,
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

    def plot_oof_confusion(self, key: Any):
        cm = self.results[key]["oof_confusion"]
        plt.figure(figsize=(5, 4))
        plt.imshow(cm.values, interpolation="nearest")
        plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
        plt.yticks(range(len(cm.index)), cm.index)
        plt.title(f"OOF confusion — region {key}")
        plt.colorbar()
        plt.tight_layout()
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

    # ---------- persistence ----------

    def save_all(self, outdir: str = "models", compress: int = 3) -> pd.DataFrame:
        """
        Persist per-region calibrated pipeline, uncalibrated backup, label encoder, and metadata.
        Returns a DataFrame manifest with paths and scores.
        """
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

            rows.append({
                "region": key,
                "best_name": r["best_name"],
                "calibrated_path": f"{base}.joblib",
                "encoder_path": f"{base}_label_encoder.joblib",
                "meta_path": f"{base}_meta.json",
                "oof_acc": scores["oof_acc"],
                "oof_f1_macro": scores["oof_f1_macro"],
            })
        return pd.DataFrame(rows)

    @staticmethod
    def load_bundle(basepath: str):
        """
        Load a saved bundle given base path WITHOUT suffix.
        Example: basepath='models/best_catboost_region1'
        Returns (pipeline, label_encoder, metadata_dict).
        """
        pipe = joblib.load(basepath + ".joblib")
        le = joblib.load(basepath + "_label_encoder.joblib")
        with open(basepath + "_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, le, meta

# ----------------- usage example -----------------
# xcleans = {"region1": df1, "region2": df2, ...}
# auto = MultiRegionBoostedClassifier(xcleans, random_state=42, verbose=True)
# auto.process_missing_class(scoring="accuracy", n_iter_per_model=25, max_splits=5, calibrate=True)
# manifest = auto.save_all(outdir="models", compress=3)
# print(manifest)
