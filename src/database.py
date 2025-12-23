from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import pandas as pd
import os

# --- KONFIGURASI PATH DATABASE (FIXED) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "smartreport.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Setup Engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODEL TABEL ---
class Laporan(Base):
    __tablename__ = "laporan_kerusakan"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    gedung = Column(String(50))
    ruangan = Column(String(50))
    jenis_kerusakan = Column(String(100))
    confidence_score = Column(Float)
    status = Column(String(20))
    deskripsi = Column(Text, nullable=True)

# --- FUNGSI CRUD ---
def init_db():
    Base.metadata.create_all(bind=engine)

# [FIX] Nama parameter disamakan dengan field tabel (jenis -> jenis_kerusakan, confidence -> confidence_score)
def create_laporan(gedung, ruangan, jenis_kerusakan, confidence_score, status, deskripsi=""):
    session = SessionLocal()
    try:
        new_report = Laporan(
            gedung=gedung, 
            ruangan=ruangan, 
            jenis_kerusakan=jenis_kerusakan,  # Sekarang cocok
            confidence_score=confidence_score, # Sekarang cocok
            status=status, 
            deskripsi=deskripsi
        )
        session.add(new_report)
        session.commit()
        session.refresh(new_report)
        return True
    except Exception as e:
        session.rollback()
        print(f"❌ Error Saving to DB: {e}")
        return False
    finally:
        session.close()

def get_all_laporan_as_df():
    try:
        return pd.read_sql("SELECT * FROM laporan_kerusakan ORDER BY timestamp DESC", engine)
    except Exception as e:
        print(f"⚠️ Error Reading DB: {e}")
        init_db()
        return pd.DataFrame(columns=["id", "timestamp", "gedung", "ruangan", "jenis_kerusakan", "confidence_score", "status", "deskripsi"])

def get_summary_stats():
    session = SessionLocal()
    try:
        total = session.query(Laporan).count()
        critical = session.query(Laporan).filter(Laporan.status == "Critical").count()
        return total, critical
    except Exception as e:
        return 0, 0
    finally:
        session.close()