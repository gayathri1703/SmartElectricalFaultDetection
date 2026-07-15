"""
utils.py
--------
Shared helper functions used across the project.

Why this file exists:
Both train.py and predict.py need to load and clean the raw dataset
in exactly the same way. Instead of copy-pasting the same cleaning
code in two places (which is a common source of bugs when the two
copies drift apart), we write it once here and import it wherever
it is needed.
"""

import pandas as pd

# The dataset stores the target as a column literally named "Output (S)".
# We rename it internally to "target" so the rest of the codebase does not
# have to worry about spaces/parentheses in a column name.
TARGET_COLUMN = "target"
RAW_TARGET_COLUMN = "Output (S)"

# These are the 6 electrical measurements that the model uses to make
# a prediction. Order matters: the API and the training code must always
# use this exact order.
FEATURE_COLUMNS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def load_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw detect_dataset.csv file and clean it.

    Cleaning steps performed:
    1. Drop the two empty "Unnamed: 7" / "Unnamed: 8" columns.
       These exist because the original CSV has two trailing commas
       at the end of every row, which pandas interprets as two extra
       columns full of NaN. They carry no information, so we drop them.
    2. Rename "Output (S)" to "target" for convenience.
    3. Drop any duplicate rows.
    4. Drop any rows that still contain missing values (safety net).

    Parameters
    ----------
    csv_path : str
        Path to detect_dataset.csv

    Returns
    -------
    pd.DataFrame
        A clean dataframe with columns:
        ['target', 'Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
    """
    df = pd.read_csv(csv_path)

    # Step 1: remove empty/unnamed columns (they are 100% NaN)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Step 2: rename target column to something easier to work with
    df = df.rename(columns={RAW_TARGET_COLUMN: TARGET_COLUMN})

    # Step 3: remove duplicate rows
    df = df.drop_duplicates()

    # Step 4: drop rows with any remaining missing values
    df = df.dropna()

    # Step 5: keep only the columns we actually need, in a fixed order
    df = df[[TARGET_COLUMN] + FEATURE_COLUMNS]

    return df.reset_index(drop=True)
