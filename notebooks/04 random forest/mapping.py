
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