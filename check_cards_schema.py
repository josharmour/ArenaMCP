
import sqlite3

db_path = r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_98812f6c01d954cdb316449ee0f0ba00.mtga"

def check_schema():
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.execute("PRAGMA table_info(Cards)")
    for row in cursor.fetchall():
        print(row)
    conn.close()

check_schema()
