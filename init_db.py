# init_db.py
from app.db import engine
from app.models import Base

def init():
    print("Creating tables in the database...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created!")

if __name__ == "__main__":
    init()
