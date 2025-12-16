import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
import psycopg2
import pandas as pd
from psycopg2 import sql
from pathlib import Path
import src.utils.data_cleaning as dc
import src.configs.config as cfg

def ingest_csv_to_postgres(csv_path:str, table_name:str, schema_name:str, db_config:dict):
    """Import CSV data into a PostgreSQL table."""

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    try:
        # Load and clean data
        df = pd.read_csv(csv_path)
        clean_df = dc.clean_dataframe(df)
        meta_df = dc.generate_metadata(clean_df)

        # Create schema
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {};")
            .format(sql.Identifier(schema_name))
        )

        # Create table
        cols_sql = sql.SQL(", ").join(
            sql.SQL("{} VARCHAR").format(sql.Identifier(c))
            for c in df.columns
        )

        cur.execute(
            sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}.{} (
                    {}
                );
            """).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                cols_sql
            )
        )

        print(f"[OK] Table ready: {schema_name}.{table_name}")

        # COPY all rows
        with open(csv_path, "r") as f:
            cur.copy_expert(
                sql.SQL(
                    "COPY {}.{} FROM STDIN WITH (FORMAT csv, HEADER true);"
                ).format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name)
                ),
                f
            )

        conn.commit()
        print("[OK] All rows imported successfully")

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":

    csv_path = "data/in/financials.csv"
    db_config = {
        "dbname": cfg.POSTGRES_USER,
        "user": cfg.POSTGRES_USER,
        "password": cfg.POSTGRES_PASSWORD,
        "host": cfg.POSTGRES_HOST,
        "port": cfg.POSTGRES_PORT,
    }

    ingest_csv_to_postgres(
        csv_path=csv_path,
        schema_name = "schema_1",
        table_name = os.path.basename(csv_path).split(".")[0],
        db_config=db_config,
    )
