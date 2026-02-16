import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database Connection
# We use the service name 'postgres' (from docker-compose-db.yml alias) or 'flight-price-db' 
# In docker-compose-app.yml, we are on 'flight_network'. 
# The hostname for the DB service in docker-compose-db.yml is 'postgres' (service name) or 'flight-price-db' (container name).
# Let's use the container name for clarity if accessible, or service name if in same compose.
# Since they are different compose files, we must use the container name or a consistent alias.
# defined in docker-compose-db.yml: container_name: flight-price-db
DB_USER = os.getenv("POSTGRES_USER", "airflow")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")
DB_HOST = os.getenv("POSTGRES_HOST", "flight-price-db") # Defaulting to container name
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "airflow")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define Table
prediction_logs = Table(
    'prediction_logs', metadata,
    Column('id', Integer, primary_key=True),
    Column('request_timestamp', DateTime, default=datetime.utcnow),
    Column('source', String),
    Column('destination', String),
    Column('airline', String),
    Column('departure_time', String),
    Column('arrival_time', String),
    Column('duration_hrs', Float),
    Column('total_stops', Integer),
    Column('seasonality', String),
    Column('days_before_departure', Integer),
    Column('aircraft_type', String),
    Column('travel_class', String),
    Column('predicted_price', Float),
    Column('model_refined_at', DateTime)
)

def init_db():
    print(f"Connecting to database at {DB_HOST}...")
    try:
        metadata.create_all(engine)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    init_db()
