import sqlite3

conn = sqlite3.connect("llm_cache.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)
for (tname,) in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tname})").fetchall()]
    print(f"\n{tname}: {count} rows, columns={cols}")
    if tname != "cost_records" and count > 0:
        row = conn.execute(f"SELECT * FROM {tname} LIMIT 1").fetchone()
        for col, val in zip(cols, row):
            v = str(val)
            print(f"  {col}: {v[:120]}")
conn.close()
