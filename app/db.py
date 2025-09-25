"""Database configuration and session utilities."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

url = make_url(DATABASE_URL)
connect_args: dict[str, Any] = {}
if url.drivername.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    database_path = url.database
    if database_path and database_path != ":memory:":
        # Normalise file URIs and ensure the directory exists before connecting.
        if database_path.startswith("file:"):
            database_path = database_path.replace("file:", "", 1)

        resolved_path = Path(database_path)
        if not resolved_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            resolved_path = (project_root / resolved_path).resolve(strict=False)
        else:
            resolved_path = resolved_path.expanduser().resolve(strict=False)

        os.makedirs(resolved_path.parent, exist_ok=True)
        path_str = resolved_path.as_posix()
        if DATABASE_URL.startswith("sqlite+pysqlite"):
            DATABASE_URL = f"sqlite+pysqlite:///{path_str}"
        else:
            DATABASE_URL = f"sqlite:///{path_str}"

        # Recompute the URL so the engine sees the absolute path.
        url = make_url(DATABASE_URL)

engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session, future=True)


def init_db() -> None:
    """Create all tables."""
    # Import models within the function to avoid circular imports.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session."""
    with session_scope() as session:
        yield session
