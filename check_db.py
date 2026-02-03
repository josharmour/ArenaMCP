
import sqlite3
import sys
from pathlib import Path

db_path = r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_98812f6c01d954cdb316449ee0f0ba00.mtga"

def check_card(grp_id):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute("SELECT * FROM Cards WHERE GrpId = ?", (grp_id,))
        row = cursor.fetchone()
        if row:
            print(f"Found card {grp_id}: {row}")
        else:
            print(f"Card {grp_id} NOT FOUND")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

check_card(98498)
check_card(83887) # Known existing card
