
import sqlite3

db_path = r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_881d59e43670b532d4292f8d5cb00c5d.mtga"

def check_loc(loc_id):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute("SELECT * FROM Localizations_enUS WHERE LocId = ?", (loc_id,))
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(f"Found in OLD DB Loc {loc_id}: {row}")
        else:
            print(f"Loc {loc_id} NOT FOUND in OLD DB")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

check_loc(457484)
