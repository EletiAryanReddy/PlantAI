import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "../data/supplements.csv")

def load_supplements():
    try:
        df = pd.read_csv(CSV_PATH)

        supplements = {}

        for _, row in df.iterrows():

            disease = str(row.get("disease_name", "")).strip()

            supplements[disease] = {
                "name": str(row.get("supplement name", "")).strip(),
                "image": str(row.get("supplement image", "")).strip(),
                "buy_link": str(row.get("buy link", "")).strip()
            }

        return supplements

    except Exception as e:
        print("CSV ERROR:", e)
        return {}
