# Standard libraries
from typing import List, Tuple

# Data manipulation and numerical computation
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, f_classif, f_regression

def drop_exact_duplicates(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop exact duplicate columns from a DataFrame.

    This function identifies and removes columns in the DataFrame that are exact duplicates 
    of other columns. Duplicate columns are those that have identical values across all rows.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.

    Returns:
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame with duplicate columns removed.
        - A list of the names of the dropped duplicate columns.

    Notes:
    -----
    - The function computes a hash signature for each column to efficiently identify duplicates.
    - If two columns have the same hash and their values are identical, one of them is dropped.

    Example Usage:
    --------------
    X, dropped_dup_cols = drop_exact_duplicates(X)
    print(f"Dropped exact duplicate columns: {dropped_dup_cols}")
    """
    sig = X.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
    seen = {}
    dup = []
    for c, h in sig.items():
        if h in seen and X[c].equals(X[seen[h]]):
            dup.append(c)
        else:
            seen[h] = c
    return X.drop(columns=dup), dup

# # Example usage:
# X, dropped_dup_cols = drop_exact_duplicates(X)
# print(f"Dropped exact duplicate columns: {dropped_dup_cols}")

def drop_high_missing(X: pd.DataFrame, thresh: float = 0.40) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with a high proportion of missing values from a DataFrame.

    This function identifies columns in the DataFrame where the proportion of missing values 
    exceeds a specified threshold and removes them. Missing values are identified as NaN 
    or the string "Unassessed", which is replaced with NaN before processing.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.
    thresh : float, optional
        The proportion threshold above which a column is considered to have high missing values 
        and is dropped. Default is 0.40 (40%).

    Returns:
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame with high-missing columns removed.
        - A list of the names of the dropped columns.

    Notes:
    -----
    - The function replaces the string "Unassessed" with NaN before calculating the proportion 
      of missing values.
    - Columns with a proportion of missing values greater than `thresh` are dropped.

    Example Usage:
    --------------
    X, dropped_cols = drop_high_missing(X, thresh=0.40)
    print(f"Dropped columns with >40% missing: {dropped_cols}")
    """
    # Replace "Unassessed" with NaN
    X = X.replace("Unassessed", np.nan)

    # Identify columns with a high proportion of missing values
    to_drop = X.columns[X.isna().mean() > thresh].tolist()

    # Drop the identified columns
    X2 = X.drop(columns=to_drop)

    return X2, to_drop


# # Example usage:
# X, dropped_cols = drop_high_missing(X, thresh=0.40)
# print(f"Dropped columns with >40% missing: {dropped_cols}")

def drop_quasi_constant_cat(
    X: pd.DataFrame, 
    cat_cols: List[str], 
    p: float = 0.99
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop quasi-constant categorical columns from a DataFrame.

    This function identifies categorical columns where the most frequent value 
    accounts for at least `p` proportion of the data and removes them from the DataFrame. 
    Quasi-constant columns are those that provide little variability and are unlikely 
    to be useful for modeling.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.
    cat_cols : List[str]
        A list of categorical column names to consider for quasi-constant checking.
    p : float, optional
        The proportion threshold above which a column is considered quasi-constant.
        Default is 0.99.

    Returns:
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame with quasi-constant categorical columns removed.
        - A list of the names of the dropped quasi-constant categorical columns.

    Notes:
    -----
    - The function ensures that only columns present in the DataFrame are processed.
    - Missing values are included in the value counts when determining the most frequent value.

    Example Usage:
    --------------
    cat_cols = X.select_dtypes(exclude=['number']).columns
    X, dropped_cat_cols = drop_quasi_constant_cat(X, cat_cols.tolist(), p=0.99)
    print(f"Dropped quasi-constant categorical columns: {dropped_cat_cols}")
    """
    dropped = []
    for c in cat_cols:
        # Calculate the normalized value counts (including NaNs)
        vc = X[c].value_counts(normalize=True, dropna=False)
        
        # Check if the most frequent value exceeds the threshold
        if len(vc) and vc.iloc[0] >= p:
            dropped.append(c)
    
    # Drop the identified quasi-constant columns
    X2 = X.drop(columns=dropped)
    return X2, dropped


# # Example usage:
# cat_cols = X.select_dtypes(exclude=['number']).columns
# X, dropped_cat_cols = drop_quasi_constant_cat(X, cat_cols.tolist(), p=0.99)
# print(f"Dropped quasi-constant categorical columns: {dropped_cat_cols}")

def drop_high_cardinality_cat(
    X: pd.DataFrame, 
    cat_cols: List[str], 
    max_unique: int = 100, 
    ratio: float = 0.50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop high-cardinality categorical columns from a DataFrame.

    This function identifies categorical columns with a high number of unique values 
    (cardinality) and removes them from the DataFrame. High-cardinality columns can 
    increase the complexity of the model and may not provide significant value.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.
    cat_cols : List[str]
        A list of categorical column names to consider for cardinality checking.
    max_unique : int, optional
        The maximum number of unique values allowed in a categorical column. 
        Default is 100.
    ratio : float, optional
        The maximum ratio of unique values to the total number of rows in the DataFrame.
        Default is 0.50.

    Returns:
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame with high-cardinality categorical columns removed.
        - A list of the names of the dropped high-cardinality categorical columns.

    Notes:
    -----
    - A column is considered high-cardinality if the number of unique values exceeds 
      `max_unique` or if the ratio of unique values to the total number of rows exceeds `ratio`.
    - The function ensures that only columns present in the DataFrame are processed.

    Example Usage:
    --------------
    cat_cols = X.select_dtypes(exclude=['number']).columns
    X, dropped_high_card_cols = drop_high_cardinality_cat(X, cat_cols.tolist(), max_unique=100, ratio=0.50)
    print(f"Dropped high-cardinality categorical columns: {dropped_high_card_cols}")
    """
    n = len(X)
    lim = min(max_unique, int(ratio * n))
    dropped = [c for c in cat_cols if c in X.columns and X[c].nunique(dropna=False) > lim]
    X2 = X.drop(columns=dropped)
    return X2, dropped

# # Example usage:
# cat_cols = X.select_dtypes(exclude=['number']).columns
# X, dropped_high_card_cols = drop_high_cardinality_cat(X, cat_cols.tolist(), max_unique=100, ratio=0.50)
# print(f"Dropped high-cardinality categorical columns: {dropped_high_card_cols}")

def drop_quasi_constant_num(
    X: pd.DataFrame, num_cols: List[str], thresh: float = 1e-5
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop quasi-constant numeric columns from a DataFrame.

    This function identifies numeric columns with very low variance (below a specified threshold)
    and removes them from the DataFrame. Quasi-constant columns are those that have almost the same
    value across all rows, which makes them uninformative for modeling.

    Parameters:
    ----------
    X : pd.DataFrame
        The input DataFrame containing the features.
    num_cols : List[str]
        A list of numeric column names to consider for variance checking.
    thresh : float, optional
        The variance threshold below which columns are considered quasi-constant.
        Default is 1e-5.

    Returns:
    -------
    Tuple[pd.DataFrame, List[str]]
        - A DataFrame with quasi-constant numeric columns removed.
        - A list of the names of the dropped quasi-constant numeric columns.

    Notes:
    -----
    - Missing values in the numeric columns are imputed using the median before calculating variance.
    - The VarianceThreshold from sklearn is used to identify quasi-constant columns.
    """
    if not num_cols:
        return X, []

    # Impute missing values with the median
    imp = SimpleImputer(strategy="median")
    Xn = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

    # Apply VarianceThreshold to identify columns with low variance
    vt = VarianceThreshold(threshold=thresh)
    vt.fit(Xn)

    # Identify kept and dropped columns
    kept = [c for c, k in zip(num_cols, vt.get_support()) if k]
    dropped = [c for c in num_cols if c not in kept]

    # Drop the quasi-constant columns from the original DataFrame
    X2 = X.drop(columns=dropped)
    return X2, dropped


# # Example usage:
# num_cols = X.select_dtypes(include=['number']).columns  # Select numeric columns
# X, dropped_num_cols = drop_quasi_constant_num(X, num_cols.tolist(), thresh=1e-5)
# print(f"Dropped quasi-constant numeric columns: {dropped_num_cols}")

def prefilter_num_univariate(
    X: pd.DataFrame, y: pd.Series, num_cols: List[str], k: int = 300
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Perform a univariate feature selection for numeric columns based on their relationship with the target variable.

    This function selects the top `k` numeric features that have the highest scores in a univariate statistical test 
    (ANOVA F-value for classification or F-statistic for regression) with respect to the target variable `y`.

    Parameters:
    ----------
    X : pd.DataFrame
        The input dataframe containing features.
    y : pd.Series
        The target variable.
    num_cols : List[str]
        A list of numeric column names to consider for feature selection.
    k : int, optional
        The number of top features to keep, by default 300.

    Returns:
    -------
    Tuple[pd.DataFrame, List[str], pd.Series]
        - A dataframe containing the top `k` numeric features.
        - A list of the names of the selected top `k` numeric features.
        - A pandas Series containing the scores of all numeric features, sorted in descending order.

    Notes:
    -----
    - If the target variable `y` is numeric and has more than 20 unique values, the function uses `f_regression`.
      Otherwise, it uses `f_classif`.
    - Missing values in the numeric columns are imputed using the median before calculating the scores.
    """
    # Filter numeric columns that exist in the dataframe
    keep = [c for c in num_cols if c in X.columns]
    if not keep:
        return X, [], pd.Series(dtype=float)

    # Impute missing values with the median
    Xi = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X[keep]),
        columns=keep, index=X.index
    )

    # Select the appropriate statistical test based on the target variable type
    if np.issubdtype(y.dtype, np.number) and y.nunique() > 20:
        scores, _ = f_regression(Xi, y.to_numpy())
    else:
        scores, _ = f_classif(Xi, y.to_numpy())

    # Create a pandas Series of scores, sort them in descending order
    s = pd.Series(scores, index=keep).fillna(0.0).sort_values(ascending=False)

    # Select the top `k` features
    top = s.index[: min(k, len(s))].tolist()

    # Return the top features, their names, and the scores
    return pd.concat([Xi[top]], axis=1), top, s

# # Example usage:
# Xnum, top_num_cols, f_scores = prefilter_num_univariate(X, y, num_cols.tolist(), k=300)

def IWANTMYXCLEAN(
    X: pd.DataFrame,
    y: pd.Series = None,
    p: float = 0.99,
    thresh_high_missing: float = 0.95,
    max_unique: int = 100,
    ratio: float = 0.50,
    num_thresh: float = 1e-5,
    k_univariate: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Clean dataframe X by applying the available filtering functions and return
    the cleaned dataframe plus a dict with lists of dropped columns.

    Parameters mirror the underlying helper functions. If y is provided,
    univariate numeric prefiltering will be run and its outputs included.

    Returns:
    - cleaned X (pd.DataFrame)
    - dict with keys: dropped_dup, dropped_missing, dropped_quasi_cat,
                      dropped_high_card_cat, dropped_quasi_num,
                      (optional) top_num_cols, f_scores
    """
    dropped = {}

    COMPLETE_X = X.copy()

    # 1) exact duplicate columns
    X, dropped_dup_cols = drop_exact_duplicates(X)
    dropped["dropped_dup"] = dropped_dup_cols
    if verbose:
        print(f"Dropped exact duplicate columns: {dropped_dup_cols}")

    # 2) high-missing columns
    X, dropped_cols = drop_high_missing(X, thresh=thresh_high_missing)
    dropped["dropped_missing"] = dropped_cols
    if verbose:
        print(f"Dropped columns with >{thresh_high_missing*100:.0f}% missing: {dropped_cols}")
        print(len(dropped_cols))

    # 3) quasi-constant categorical
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    X, dropped_cat_cols = drop_quasi_constant_cat(X, cat_cols, p=p)
    dropped["dropped_quasi_cat"] = dropped_cat_cols
    if verbose:
        print(f"Dropped quasi-constant categorical columns: {dropped_cat_cols}")

    # 4) high-cardinality categorical
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    X, dropped_high_card_cols = drop_high_cardinality_cat(X, cat_cols, max_unique=max_unique, ratio=ratio)
    dropped["dropped_high_card_cat"] = dropped_high_card_cols
    if verbose:
        print(f"Dropped high-cardinality categorical columns: {dropped_high_card_cols}")

    # 5) quasi-constant numeric
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X, dropped_num_cols = drop_quasi_constant_num(X, num_cols, thresh=num_thresh)
    dropped["dropped_quasi_num"] = dropped_num_cols
    if verbose:
        print(f"Dropped quasi-constant numeric columns: {dropped_num_cols}")

    # 6) optional: univariate numeric prefilter (if y provided)
    if y is not None and len(num_cols):
        Xnum, top_num_cols, f_scores = prefilter_num_univariate(X, y, num_cols, k=k_univariate)
        dropped["top_num_cols"] = top_num_cols
        dropped["f_scores"] = f_scores
        if verbose:
            print(f"Selected top {len(top_num_cols)} numeric features (univariate): {top_num_cols}")
    else:
        dropped["top_num_cols"] = []
        dropped["f_scores"] = None

    # sum the value per row of the dropped columns
    uncommon_taxons = COMPLETE_X[dropped_cols].sum(axis=1)
    uncommon_taxons = uncommon_taxons.rename("Uncommon_Taxons")
    X = pd.concat([X, uncommon_taxons], axis=1)

    quasiconstants = COMPLETE_X[dropped_num_cols].sum(axis=1)
    quasiconstants = quasiconstants.rename("QuasiConstant_Numeric")
    X = pd.concat([X, quasiconstants], axis=1)

    return X