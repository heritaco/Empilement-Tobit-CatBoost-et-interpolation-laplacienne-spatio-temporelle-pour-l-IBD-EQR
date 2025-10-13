import pandas as pd
import numpy as np

def add_tol_to_ranges(tol=0.1) -> pd.DataFrame:
    ranges = pd.read_csv("..\\..\\data\\processed\\ibd_eqr_ranges_by_herlvl1_continuous_midpoint.csv")
    ranges.loc[ranges['IBD_EQR_Status'] == 'Bad', 'IBD_min'] = 0 - tol
    ranges.loc[ranges['IBD_EQR_Status'] == 'High', 'IBD_max'] = 20 + tol
    return ranges

def to_status(yhat: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column 'IBD_EQR_Status_Predicted' to the `yhat` DataFrame by mapping the predicted IBD values 
    ('IBD_Predicted') into the appropriate bin defined by the [IBD_min, IBD_max) intervals in the `ranges` DataFrame 
    for the corresponding 'HERlvl1Name'. The topmost bin per region also includes its right endpoint.

    Parameters:
    ----------
    yhat : pd.DataFrame
        A DataFrame containing the predicted IBD values ('IBD_Predicted') and the corresponding 'HERlvl1Name'.
    ranges : pd.DataFrame
        A DataFrame containing the bin definitions for each 'HERlvl1Name', including columns:
        - 'HERlvl1Name': The region name.
        - 'IBD_EQR_Status': The status corresponding to the bin.
        - 'IBD_min': The lower bound of the bin (inclusive).
        - 'IBD_max': The upper bound of the bin (exclusive, except for the topmost bin).

    Returns:
    -------
    pd.DataFrame
        A copy of the `yhat` DataFrame with an additional column 'IBD_EQR_Status_Predicted', which contains the 
        mapped status for each prediction.

    Notes:
    -----
    - The function performs a cartesian merge between `yhat` and `ranges` based on 'HERlvl1Name'.
    - Each predicted value is matched to the bin where it falls within the [IBD_min, IBD_max) interval.
    - For the topmost bin in each region, the right endpoint (IBD_max) is included.
    - In case of ties (multiple bins matching a prediction), the first match is kept.
    """
    out = yhat.copy()
    out['__ix__'] = np.arange(len(out))

    # Copy ranges and calculate the maximum right endpoint for each region
    r = ranges[['HERlvl1Name', 'IBD_EQR_Status', 'IBD_min', 'IBD_max']].copy()
    r['__max_right__'] = r.groupby('HERlvl1Name')['IBD_max'].transform('max')

    # Cartesian merge by region, then keep the single interval that matches each prediction
    m = out.merge(r, on='HERlvl1Name', how='left')

    # Check if predictions fall within the bin intervals
    pred = m['IBD_Predicted'].astype(float)
    left_ok  = pred >= m['IBD_min']
    right_ok = (pred <  m['IBD_max']) | ((pred == m['IBD_max']) & (m['IBD_max'].eq(m['__max_right__'])))
    m = m[left_ok & right_ok]

    # In case of any ties, keep the first match; then map back to original rows
    m = m.sort_values(['__ix__', 'IBD_min', 'IBD_max']).drop_duplicates('__ix__', keep='first')
    status = m.set_index('__ix__')['IBD_EQR_Status']

    # Map the status back to the original DataFrame
    out['IBD_EQR_Status_Predicted'] = out['__ix__'].map(status)
    out = out.drop(columns='__ix__')
    return out

def get_results(yhat, ranges, output_file="IBD_EQR_Status_predictions_5663.csv") -> pd.Series:
    yhat2 = to_status(yhat, ranges)
    send_predictions = yhat2['IBD_EQR_Status_Predicted']
    send_predictions = send_predictions.to_frame()
    send_predictions = send_predictions.rename(columns={'IBD_EQR_Status_Predicted': 'IBD_EQR_Status'})
    send_predictions.to_csv(output_file, index=True)
    print(f"Saved predictions to {output_file}")
    return send_predictions

def process_predictions(yhat: pd.DataFrame, ranges: pd.DataFrame, output_file="IBD_EQR_Status_predictions.csv") -> pd.Series:
    """
    Processes predictions by adding EQR status, comparing with an example dataset, and saving the results.

    Parameters:
    ----------
    yhat : pd.DataFrame
        DataFrame containing predicted IBD values and corresponding HERlvl1Name.
    ranges : pd.DataFrame
        DataFrame containing bin definitions for mapping IBD values to EQR status.
    read_example : pd.DataFrame
        DataFrame containing example predictions for comparison.
    output_file : str
        Path to save the processed predictions as a CSV file.

    Returns:
    -------
    pd.Series
        Processed predictions with the name 'IBD_EQR_Status'.
    """

    read_example = pd.read_csv("..\\..\\results\\000 example\\random_predictions.csv")
    # Add EQR status to predictions
    yhat2 = to_status(yhat, ranges)
    send_predictions = yhat2['IBD_EQR_Status_Predicted']

    # Inner join to compare with example predictions
    innerjoin = send_predictions.to_frame().merge(
        read_example.set_index('SamplingOperations_code'),
        left_index=True,
        right_index=True,
        how='inner',
        suffixes=('_predicted', '_example')
    )

    # Extract the predicted status
    truesend = innerjoin["IBD_EQR_Status_Predicted"]

    # Rename and save to CSV
    truesend = truesend.rename("IBD_EQR_Status")
    truesend = truesend.to_frame()
    truesend.to_csv(output_file)
    print(f"Saved predictions to {output_file}")

    return truesend