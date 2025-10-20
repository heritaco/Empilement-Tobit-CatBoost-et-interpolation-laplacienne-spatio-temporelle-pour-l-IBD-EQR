import pandas as pd
from typing import Optional
import numpy as np

def assign_group(df):
    """
    Requires a df with a column "HERlvl1Name".
    Adds a "Group" column based on predefined region groups.
    """
    import numpy as np
    df = df.copy()
    group1 = ["ARMORICAIN", "ARDENNES", "DEPRESSIONS SEDIMENTAIRES"]
    group2 = ["ALPES INTERNES", "PYRENEES", "PREALPES DU SUD", "JURA-PREALPES DU NORD"]
    group3 = ["MEDITERRANEEN", "COTES CALCAIRES EST", "TABLES CALCAIRES", "ALSACE", "COTEAUX AQUITAINS", "CAUSSES AQUITAINS", "DEPOTS ARGILO SABLEUX", "GRANDS CAUSSES"]
    group4 = ["MASSIF CENTRAL NORD", "VOSGES", "PLAINE SAONE", "MASSIF CENTRAL SUD", "CEVENNES", "CORSE"]
    group5 = ['LANDES']

    df["Group"] = df.apply(lambda row: 'Group 1' if row["HERlvl1Name"] in group1 else np.nan, axis=1)
    df["Group"] = df.apply(lambda row: 'Group 2' if row["HERlvl1Name"] in group2 else row["Group"], axis=1)
    df["Group"] = df.apply(lambda row: 'Group 3' if row["HERlvl1Name"] in group3 else row["Group"], axis=1)
    df["Group"] = df.apply(lambda row: 'Group 4' if row["HERlvl1Name"] in group4 else row["Group"], axis=1)
    df["Group"] = df.apply(lambda row: 'Group 5' if row["HERlvl1Name"] in group5 else row["Group"], axis=1)
    print("Assigned groups based on HERlvl1Name.")
    return df

def add_tol_to_ranges(
    ranges: pd.DataFrame | str,
    *,
    status_col: str,
    min_col: str,
    max_col: str,
    bad_label: str = "Bad",
    high_label: str = "High",
    min_floor: float = 0.0,
    max_ceiling: float = 20.0,
    tol: float = 0.1,
) -> pd.DataFrame:
    """
    Apply tolerance to boundary categories without assuming column names.

    If `ranges` is a path, read CSV first.
    Sets min(bad)=min_floor - tol and max(high)=max_ceiling + tol.
    """
    r = pd.read_csv(ranges) if isinstance(ranges, str) else ranges.copy()

    # Ensure numeric
    r[min_col] = pd.to_numeric(r[min_col], errors="coerce")
    r[max_col] = pd.to_numeric(r[max_col], errors="coerce")

    r.loc[r[status_col] == bad_label,  min_col] = min_floor - tol
    r.loc[r[status_col] == high_label, max_col] = max_ceiling + tol
    return r

def get_ranges(
    tol : float = 0.1):
    ranges = pd.read_csv("data/processed/ibd_eqr_ranges_by_herlvl1_continuous_midpoint.csv")
    return add_tol_to_ranges(
        ranges,
        status_col='IBD_EQR_Status',
        min_col='IBD_min',
        max_col='IBD_max',
        tol = tol
    )

def get_ranges2():
    return pd.read_csv("data/processed/IBD_EQR_Status_ranges_by_GROUP_Continuous.csv")


def to_status(
    yhat: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    pred_col: str,          # name of prediction column in yhat
    region_col: str = 'HERlvl1Name',  # common key to join on; None => single global region
    status_col: str = "IBD_EQR_Status",        # status label column in ranges (e.g., "IBD_EQR_Status")
    min_col: str = "IBD_min",           # left endpoint col in ranges
    max_col: str = "IBD_max",           # right endpoint col in ranges
    out_status_col: str = "Status_Predicted",
    closed_left: bool = True,
    rightmost_closed: bool = True,
) -> pd.DataFrame:
    """
    Map numeric predictions to status by interval lookup per region.
    - Intervals are [min, max) by default.
    - If rightmost_closed=True, the largest max in each region is closed (inclusive).
    - If region_col is None, all rows are treated as one region.
    """
    out = yhat.copy()
    out["__ix__"] = np.arange(len(out))

    # Prepare ranges
    req_cols = [status_col, min_col, max_col]
    r = ranges[([region_col] if region_col else []) + req_cols].copy()

    # Single-region mode if no region column
    if region_col is None:
        region_col = "__ALL_REGION__"
        out[region_col] = "__ALL__"
        r[region_col] = "__ALL__"

    # Max right endpoint per region to close only the rightmost bin
    r["__max_right__"] = r.groupby(region_col)[max_col].transform("max")

    # Merge by region (cartesian over bins of that region)
    m = out.merge(r, on=region_col, how="left", copy=False)

    # Numeric predictions
    pred = pd.to_numeric(m[pred_col], errors="coerce")

    # Bound checks
    left_ok = pred >= m[min_col] if closed_left else pred > m[min_col]
    right_closed_mask = (pred <= m[max_col]) & (m[max_col].eq(m["__max_right__"])) if rightmost_closed else (pred <= m[max_col])
    right_semi_open_mask = pred < m[max_col]
    right_ok = right_semi_open_mask | right_closed_mask

    m = m[left_ok & right_ok]

    # Keep the first match per original row if multiple intervals overlap
    m = (
        m.sort_values(["__ix__", min_col, max_col])
         .drop_duplicates("__ix__", keep="first")
    )

    # Map status back
    status_map = m.set_index("__ix__")[status_col]
    out[out_status_col] = out["__ix__"].map(status_map)
    out = out.drop(columns="__ix__")
    return out


def get_results(
    yhat: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    pred_col: str,
    region_col: Optional[str] = None,
    status_col: str = "IBD_EQR_Status",
    min_col: str = "IBD_min",
    max_col: str = "IBD_max",
    out_status_col: str = "IBD_EQR_Status_Predicted",
    output_file: str = "IBD_EQR_Status_predictions.csv",
    index: bool = True,
) -> pd.DataFrame:
    """
    Produce status predictions and write CSV. Column names are parameters.
    """
    yhat2 = to_status(
        yhat, ranges,
        pred_col=pred_col,
        region_col=region_col,
        status_col=status_col,
        min_col=min_col,
        max_col=max_col,
        out_status_col=out_status_col,
    )
    send = yhat2[[out_status_col]].rename(columns={out_status_col: status_col})
    send.to_csv(output_file, index=index)
    print(f"Saved predictions to {output_file}")
    return send

def get_results2(
    yhat: pd.DataFrame,
    ranges: pd.DataFrame,
    *,
    pred_col: str,
    region_col: Optional[str] = "Group",
    status_col: str = "IBD_EQR_Status",
    min_col: str = "IBD_min",
    max_col: str = "IBD_max",
    out_status_col: str = "IBD_EQR_Status_Predicted",
    output_file: str = "IBD_EQR_Status_predictions.csv",
    index: bool = True,
    fast: bool = True,
) -> pd.DataFrame:
    """
    Produce status predictions and write CSV. Column names are parameters.
    """
    dftest = pd.read_parquet("data/processed/taxones_pressure_predict.parquet")
    dfgroups = assign_group(dftest).copy()
    yhat = yhat.merge(dfgroups[["SamplingOperations_code", "Group"]], how="left", left_on="SamplingOperations_code", right_on="SamplingOperations_code")

    dfgroups = None
    yhat2 = to_status(
        yhat, ranges,
        pred_col=pred_col,
        region_col=region_col,
        status_col=status_col,
        min_col=min_col,
        max_col=max_col,
        out_status_col=out_status_col,
    )
    send = yhat2[["SamplingOperations_code", out_status_col]].rename(columns={out_status_col: status_col}).set_index("SamplingOperations_code")
    # rename the column to IBD_EQR_Status
    send = send.rename(columns={out_status_col: "IBD_EQR_Status"})
    send.to_csv(output_file, index=True)
    print(f"Saved predictions to {output_file}")
    return send