"""
base_full2 = dict(
    task_type="GPU",
    devices="0",
    loss_function="RMSE", # or 
    iterations=10000,
    learning_rate=0.02,
    depth=10,
    l2_leaf_reg=7.0,
    random_strength=1.0,
    bagging_temperature=0.8,
    # rsm=0.8,                  # alias de colsample_bylevel en GPU
    grow_policy="Lossguide",  # mejor con muchas columnas
    min_data_in_leaf=32,
    max_bin=128,
    verbose=200,
    random_seed=42,
    od_type="Iter",
    od_wait=150,
    eval_metric="RMSE"
)

model, scored_df = fit_catboost_and_fill(
    full,
    base=base_full2,
)
"""


import numpy as np
import pandas as pd
from typing import Tuple, Optional
from catboost import CatBoostRegressor

from catboost.utils import get_gpu_device_count
print("GPUs visible to CatBoost:", get_gpu_device_count())

def fit_catboost_and_fill(
    cleandf: pd.DataFrame,
    target: str = "IBD",
    base: Optional[dict] = None,         # e.g., {"loss_function":"RMSE","iterations":1500,"learning_rate":0.03,"depth":8}
    cat_params: Optional[dict] = None,
    use_ohe: bool = False                # False = native categoricals (recommended)
) -> Tuple[CatBoostRegressor, pd.DataFrame]:
    """
    Train on all rows with known target, then predict target for rows with NaN.
    Drops ['IBD','IBD_EQR','IBD_EQR_Status'] from features if present.
    Returns (fitted_model, scored_df) where scored_df = rows with NaN target + target_pred.
    """

    # 1) split into train/score by target presence
    df_train = cleandf[cleandf[target].notna()].copy()
    df_score = cleandf[cleandf[target].isna()].copy()

    # 2) drop leakage columns
    drop_cols = [c for c in ["IBD", "IBD_EQR", "IBD_EQR_Status"] if c in cleandf.columns]

    X = df_train.drop(columns=drop_cols + [target], errors="ignore")
    y = df_train[target].astype(float)

    # 3) detect dtypes
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 4) light preprocessing
    # numeric: leave NaN (CatBoost handles); categorical: fill with "(missing)"
    X_num = X[num_cols].copy()
    X_cat = X[cat_cols].copy().fillna("(missing)")

    if use_ohe:
        X_cat = pd.get_dummies(X_cat, drop_first=False)
        X_tr_proc = pd.concat([X_num.reset_index(drop=True),
                               X_cat.reset_index(drop=True)], axis=1)
        cat_features = None
    else:
        X_tr_proc = pd.concat([X_num.reset_index(drop=True),
                               X_cat.reset_index(drop=True)], axis=1)
        cat_features = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    # 5) params and fit on ALL training data (no eval_set, no early stopping)
    base_params = base
    if cat_params:
        base_params.update(cat_params)

    model = CatBoostRegressor(**base_params)

    fit_kwargs = dict(X=X_tr_proc, y=y, use_best_model=False)
    if not use_ohe:
        fit_kwargs["cat_features"] = cat_features
    model.fit(**fit_kwargs)

    # 6) score NaN-target rows
    if df_score.empty:
        scored_df = pd.DataFrame(columns=list(cleandf.columns) + [f"{target}_pred"])
        return model, scored_df

    Xs = df_score.drop(columns=drop_cols + [target], errors="ignore")
    Xs_num = Xs[num_cols].copy()
    Xs_cat = Xs[cat_cols].copy().fillna("(missing)")

    if use_ohe:
        Xs_cat = pd.get_dummies(Xs_cat, drop_first=False)
        Xs_cat = Xs_cat.reindex(columns=X_cat.columns, fill_value=0)
        Xs_proc = pd.concat([Xs_num.reset_index(drop=True),
                             Xs_cat.reset_index(drop=True)], axis=1)
    else:
        Xs_proc = pd.concat([Xs_num.reset_index(drop=True),
                             Xs_cat.reset_index(drop=True)], axis=1)

    df_score[f"{target}_pred"] = model.predict(Xs_proc)
    scored_df = df_score

    return model, scored_df