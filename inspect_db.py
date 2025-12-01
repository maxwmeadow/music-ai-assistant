"""
Quick script to inspect the database schema
"""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "backend/training_data.db"

print(f"Inspecting: {db_path}")
print("=" * 60)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("\nTables:")
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "=" * 60)

# Show schema for each table
for table in tables:
    table_name = table[0]
    print(f"\nSchema for '{table_name}':")
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    for col in columns:
        col_id, name, type_, notnull, default, pk = col
        print(f"  {name:20s} {type_:15s} {'NOT NULL' if notnull else ''} {'PRIMARY KEY' if pk else ''}")

    # Show sample data
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
    sample = cursor.fetchone()
    if sample:
        print(f"\n  Sample row:")
        for i, val in enumerate(sample):
            col_name = columns[i][1]
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            print(f"    {col_name}: {val_str}")

conn.close()