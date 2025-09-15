from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db import Base

class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(Integer, ForeignKey("events.id"), index=True)
    uri: Mapped[str] = mapped_column(String(1024))
    thumb_uri: Mapped[str] = mapped_column(String(1024))
    embedding_path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    file_hash = Column(String, unique=True, index=True)