import sqlite3
import os

DB_FILE = "hybrid_autolegal.db"

def init_db():
    # remove old db if corrupted
    if not os.path.exists(DB_FILE):
        print("📂 Creating new database...")
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # --- Users ---
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")

    # --- FAQs ---
    cur.execute("""CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )""")

    # --- Clauses ---
    cur.execute("""CREATE TABLE IF NOT EXISTS clauses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original TEXT,
        simplified TEXT
    )""")

    conn.commit()
    conn.close()
    print("✅ Tables created successfully.")


def seed_data():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Add admin user (only if not exists)
    cur.execute("SELECT * FROM users WHERE username=?", ("admin",))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "admin123"))
        print("👤 Default admin user created (username=admin, password=admin123)")

    # Add sample FAQs
    sample_faqs = [
        ("What is a contract?", "A contract is a legally binding agreement between two or more parties."),
        ("What is an affidavit?", "An affidavit is a written sworn statement of fact, signed before an authority."),
    ]
    cur.executemany("INSERT INTO faq (question, answer) VALUES (?, ?)", sample_faqs)
    print(f"📘 Inserted {len(sample_faqs)} sample FAQs.")

    # Add sample Clauses
    sample_clauses = [
        ("The lessee shall indemnify the lessor against all claims.", "The tenant must protect the landlord from any claims."),
        ("This agreement shall terminate immediately upon breach of contract.", "The deal ends right away if either side breaks the terms."),
    ]
    cur.executemany("INSERT INTO clauses (original, simplified) VALUES (?, ?)", sample_clauses)
    print(f"📜 Inserted {len(sample_clauses)} sample clauses.")

    conn.commit()
    conn.close()
    print("🎉 Database populated successfully!")


if __name__ == "__main__":
    init_db()
    seed_data()

