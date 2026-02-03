
import sqlite3

db_path = r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_98812f6c01d954cdb316449ee0f0ba00.mtga"

def check_loc_base(loc_id):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.execute("SELECT * FROM Localizations WHERE LocId = ?", (loc_id,))
        rows = cursor.fetchall()
        if rows:
            for row in rows:
                print(f"Found in Localizations {loc_id}: {row}")
        else:
            print(f"Loc {loc_id} NOT FOUND in Localizations")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

check_loc_base(457484)
