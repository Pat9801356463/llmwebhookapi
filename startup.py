# startup.py
from engine import db

if __name__ == "__main__":
    print("🚀 Running startup tasks...")
    try:
        db.create_tables()
        print("✅ Tables ensured in Postgres.")
    except Exception as e:
        print(f"❌ Error ensuring tables: {e}")
