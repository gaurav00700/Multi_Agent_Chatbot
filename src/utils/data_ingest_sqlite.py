import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)
from pathlib import Path
import pandas as pd
import sqlite3
import json
import src.utils.data_cleaning as dc
import src.configs.config as cfg

# Constants
DEFAULT_DB_PATH = Path("data/temp/ingested.db")
DEFAULT_TABLE_NAME = "data_table"
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".json", ".xls", ".xlsx"}

def persist_to_sqlite(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    db_path: Path,
    table_name: str
    ):
    """Save pandas dataframe and metadata to SQLite.

    Args:
        df (pd.DataFrame): DataFrame to save
        meta_df (pd.DataFrame): Metadata DataFrame to save
        db_path (Path): Path to SQLite database
        table_name (str): Name of the table to create

    Returns:
        None
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    meta_df.to_sql(f"{table_name}__meta", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def read_csv_or_txt(path: Path) -> pd.DataFrame:
    """
    Read CSV or TXT file.
    TXT is assumed to be delimiter-based (comma, tab, or pipe auto-detected).
    """
    return pd.read_csv(path, sep=None, engine="python")

def read_json(path: Path) -> pd.DataFrame:
    """
    Read JSON file.
    Supports:
    - list of dicts
    - dict of lists
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)

def read_excel(path: Path) -> pd.DataFrame:
    """
    Read Excel file.
    Defaults to first sheet.
    """
    return pd.read_excel(path)

def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Dispatch loader based on file extension.
    """
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return read_csv_or_txt(file_path)
    elif ext == ".txt":
        return read_csv_or_txt(file_path)
    elif ext == ".json":
        return read_json(file_path)
    elif ext in {".xls", ".xlsx"}:
        return read_excel(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

def ingest_file_sqlite(
    file_path: str,
    db_path: str = str(DEFAULT_DB_PATH),
    table_name: str = DEFAULT_TABLE_NAME,
    ):
    """
    Ingest CSV, TXT, JSON, or Excel file into SQLite with cleaning + metadata.

    Args:
        file_path (str): Path to input file
        db_path (str): SQLite DB path
        table_name (str): SQLite table name

    Returns:
        clean_df, meta_df
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Load
    df = load_dataframe(file_path)

    # Clean + metadata
    clean_df = dc.clean_dataframe(df)
    meta_df = dc.generate_metadata(clean_df)

    # Persist
    persist_to_sqlite(
        clean_df,
        meta_df,
        Path(db_path),
        table_name
    )

    print(f"[OK] Ingested {file_path.name}")
    print(f"Table: {table_name}")
    print(f"Rows: {len(clean_df)}")
    print("Columns:", list(clean_df.columns))

    return clean_df, meta_df

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Ingest a csv | txt | json | xls | xlsx into the sqlite db")
    parser.add_argument("--file_path", type=str, default="data/in/financials.csv", help="Path to a csv | txt | json | xls | xlsx  file to ingest")
    args = parser.parse_args()

    ingest_file_sqlite(
        file_path=args.file_path,  # csv | txt | json | xls | xlsx
        db_path="data/temp/ingested.db",
        table_name="financials"
    )
