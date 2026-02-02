from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List
from app.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relacionamento
    sessions: Mapped[List["TrainingSession"]] = relationship(back_populates="user")


class TrainingSession(Base):
    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # O que o aluno tocou (ex: "C,E,G")
    input_notes: Mapped[str] = mapped_column(String(255))
    # O que a IA respondeu
    predicted_chord: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="sessions")