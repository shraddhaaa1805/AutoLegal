import sqlite3
import csv
import os

# Path to your database
DB_PATH = "hybrid_autolegal.db"

# Path to your CSV
CSV_PATH = os.path.join("legal_data", "laws.csv")

# Connect to database
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS laws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    law_type TEXT,
    original TEXT,
    simplified TEXT
)''')

# Read and insert CSV content
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        c.executemany("INSERT INTO laws (law_type, original, simplified) VALUES (?, ?, ?)", reader)
        print("✅ All laws imported successfully from laws.csv!")
else:
    print("⚠️ laws.csv not found! Please place it inside the 'legal_data' folder.")

# Commit and close
conn.commit()
conn.close()
print("🎉 Database populated successfully!")
