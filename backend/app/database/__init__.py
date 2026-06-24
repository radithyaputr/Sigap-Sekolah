"""
SIGAP Sekolah - Database Configuration
SQLAlchemy 2.0 async with PostgreSQL.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from datetime import datetime
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://sigap:sigap123@localhost:5432/sigap_sekolah"
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), default="guru")  # admin, guru, kepala_sekolah
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    nisn = Column(String(50), unique=True, index=True, nullable=False)
    nama = Column(String(255), nullable=False)
    kelas = Column(String(50), nullable=False)
    jurusan = Column(String(100))
    angkatan = Column(Integer)
    nama_orang_tua = Column(String(255))
    pekerjaan_orang_tua = Column(String(100))
    no_hp = Column(String(50))
    alamat = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class RiskAnalysis(Base):
    __tablename__ = "risk_analyses"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    persen_mapel_tuntas = Column(Float, nullable=False)
    nilai_produktif = Column(Float, nullable=False)
    kehadiran = Column(Float, nullable=False)
    alpha_hari = Column(Integer, nullable=False)
    prestasi_lomba = Column(Integer, nullable=False)
    pelanggaran_berat = Column(Integer, nullable=False)
    nilai_sikap = Column(Float, nullable=False)
    lulus_ukk = Column(Integer, nullable=False)

    risk_score = Column(Float, nullable=False)
    risk_category = Column(String(50), nullable=False)
    violations = Column(JSON, nullable=False)
    recommendations = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BatchAnalysis(Base):
    __tablename__ = "batch_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    total_students = Column(Integer, nullable=False)
    processed = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    results_summary = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100))
    resource_id = Column(Integer)
    details = Column(JSON)
    ip_address = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    description = Column(Text)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency injection for database sessions."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
