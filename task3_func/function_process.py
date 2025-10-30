# process.py
import os, io, json, datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from azure.storage.blob import BlobServiceClient

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_nutritional_data_from_azurite():
    load_dotenv()  # read .env if present

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("DATASET_CONTAINER", "datasets")
    blob_name = os.getenv("DATASET_BLOB", "All_Diets.csv")
    nosql_dir = Path(os.getenv("NOSQL_DIR", "simulated_nosql"))
    nosql_file = os.getenv("NOSQL_FILE", "results.json")

    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set")

    print(f"[{_now()}] Connecting to Azurite…")
    bsc = BlobServiceClient.from_connection_string(conn_str)
    container = bsc.get_container_client(container_name)
    blob = container.get_blob_client(blob_name)

    print(f"[{_now()}] Downloading blob: {container_name}/{blob_name}")
    raw = blob.download_blob().readall()

    print(f"[{_now()}] Reading CSV into DataFrame…")
    df = pd.read_csv(io.BytesIO(raw))

    # Normalize column names just in case (trim & exact match)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["Diet_type", "Protein(g)", "Carbs(g)", "Fat(g)"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    print(f"[{_now()}] Calculating averages per Diet_type…")
    avg = (
        df.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]]
        .mean(numeric_only=True)
        .round(2)
        .reset_index()
    )

    # Build JSON “document”
    result_doc = {
        "source": f"{container_name}/{blob_name}",
        "processed_at": _now(),
        "stats": avg.to_dict(orient="records"),
    }

    # Simulated NoSQL write
    nosql_dir.mkdir(parents=True, exist_ok=True)
    out_path = nosql_dir / nosql_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, ensure_ascii=False, indent=2)

    print(f"[{_now()}] Wrote results to {out_path.resolve()}")
    return "OK"

if __name__ == "__main__":
    try:
        status = process_nutritional_data_from_azurite()
        print(f"[{_now()}] Function completed: {status}")
    except Exception as e:
        print(f"[{_now()}] ERROR: {e}")
        raise
