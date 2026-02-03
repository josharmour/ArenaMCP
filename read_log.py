
import os

log_file = r"z:\ArenaMCP\debug_log.txt"
try:
    with open(log_file, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        # Read last 10kb
        read_size = min(file_size, 10240)
        f.seek(-read_size, os.SEEK_END)
        content = f.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()
        for line in lines[-20:]:
            print(line)
except Exception as e:
    print(f"Error reading log: {e}")
