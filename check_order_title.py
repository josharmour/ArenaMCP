
import sqlite3

db_path = r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_98812f6c01d954cdb316449ee0f0ba00.mtga"

def check_order_title(grp_id):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        # Check if Order_Title exists first
        cursor = conn.execute("SELECT Order_Title FROM Cards WHERE GrpId = ?", (grp_id,))
        row = cursor.fetchone()
        if row:
            print(f"Order_Title for {grp_id}: {row[0]}")
        else:
            print(f"Card {grp_id} NOT FOUND")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

check_order_title(98498)
check_order_title(83887)
