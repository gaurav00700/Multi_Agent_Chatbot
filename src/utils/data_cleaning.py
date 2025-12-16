import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
import pandas as pd
import re
import unicodedata
import json
from pathlib import Path
import src.configs.config as cfg

NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-"}

# Helpers
def normalize_colname(name: str) -> str:
    """Make column names SQL-safe and consistent.

    Args:
        name: The input column name to be normalized.

    Returns:
        A normalized, SQL-safe column name in lowercase with underscores.
        Returns "column" if the input is empty or invalid.
    """
    # Normalize Unicode characters to their base form with combining marks
    name = unicodedata.normalize("NFKD", str(name))

    # Convert to ASCII, ignoring non-ASCII characters
    name = name.encode("ascii", errors="ignore").decode("ascii")

    # Convert to lowercase and remove leading/trailing whitespace
    name = name.lower().strip()

    # Replace any non-alphanumeric characters with underscores ("First-Name" -> "First_Name")
    name = re.sub(r"\W+", "_", name)

    # Remove leading/trailing underscores and return "column" if empty
    return name.strip("_") or "column"

def normalize_column_names(columns):
    """Normalize and deduplicate column names to ensure uniqueness.

    When multiple columns have the same normalized name, this function
    appends a numeric suffix to make them unique while preserving the original
    normalized format.

    Args:
        columns: List of column names to be normalized and deduplicated.

    Returns:
        List of normalized column names with unique identifiers appended
        where necessary to ensure all names are unique.
    """
    seen = {}
    result = [] # final normalized column names

    # Process each column in the input list
    for c in columns:        
        base = normalize_colname(c) # Normalize the column name
        count = seen.get(base, 0)   # Default to 0 if not seen
        seen[base] = count + 1  # Update the count

        # If this is the first time we've seen this normalized name,
        # use it as-is. Otherwise, append the count as a suffix
        if count == 0:
            result.append(base)
        else:
            result.append(f"{base}_{count}")

    return result

def handle_missing_values(df: pd.DataFrame, small_data_threshold: int = 100) -> pd.DataFrame:
    """Handle missing values with different strategies based on dataset size.

    Args:
        df: Input DataFrame containing missing values
        small_data_threshold: Threshold (in rows) to determine if dataset is small.Defaults to 100.

    Returns:
        DataFrame with missing values handled according to dataset size:
        - For small datasets (< threshold): Missing values are filled (numeric: median, categorical: mode or empty string)
        - For large datasets (≥ threshold): Rows with missing values are dropped
    """
    df = df.copy()
    n_rows = len(df)

    if n_rows < small_data_threshold:
        # Fill missing values
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                # Use median if available, else mean
                if df[c].notna().any():
                    df[c] = df[c].fillna(df[c].median())
            else:
                # Categorical / text: fill with mode or empty string
                if df[c].notna().any():
                    df[c] = df[c].fillna(df[c].mode().iloc[0])
                else:
                    df[c] = df[c].fillna("")
    else:
        # Large dataset: drop rows with any missing values
        df = df.dropna()

    return df

# Core logic
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a pandas DataFrame by performing several data quality operations.

    This function performs a series of cleaning operations on a DataFrame to:
    1. Standardize column names
    2. Clean null-like values
    3. Convert columns to numeric where possible
    4. Remove duplicate rows
    5. Handle missing values based on dataset size

    Args:
        df: Input pandas DataFrame to be cleaned.

    Returns:
        Cleaned pandas DataFrame with standardized column names and improved data quality.
    """
    df = df.copy()

    # Standardize Column Names
    df.columns = normalize_column_names(df.columns)

    # Clean Null-like Values
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)  # Convert all values to strings
            .str.strip()  # Remove leading/trailing whitespace
            .replace({t: None for t in NULL_TOKENS})  # Replace null-like tokens with None
        )

    # Convert Columns to Numeric (Safe Conversion)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    # Remove Duplicate Rows
    df = df.drop_duplicates()

    # Handle missing values
    df = handle_missing_values(df, small_data_threshold=100)

    return df

def generate_metadata(df: pd.DataFrame):
    """Generate lightweight metadata about a DataFrame for agent awareness.

    This function creates a metadata summary of a DataFrame that includes:
    - Column names
    - Data types
    - Percentage of null values in each column
    - Sample values from each column

    Args:
        df: Input pandas DataFrame to analyze.

    Returns:
        DataFrame containing metadata about each column in the input DataFrame.
        Each row represents one column from the original DataFrame with its metadata.
    """
    meta = []

    # Iterate through each column in the input DataFrame
    for c in df.columns:
        # Create a dictionary containing metadata for the current column
        meta.append({
            "column_name": c,  # The name of the column
            "dtype": str(df[c].dtype),  # Convert dtype to string for consistency
            "null_ratio": float(df[c].isna().mean()),  # Calculate percentage of null values
            "example_values": json.dumps(       # Sample values from the column
                df[c].dropna().head(3).tolist()
            )
        })

    return pd.DataFrame(meta)