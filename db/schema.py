"""SQLAlchemy models describing our application schema."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    dataset_type: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    examples: Mapped[int] = mapped_column(Integer, nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="dataset"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    base_model: Mapped[str] = mapped_column(String(128), nullable=False)
    adapter_path: Mapped[Optional[str]] = mapped_column(String(256))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dataset: Mapped[Dataset] = relationship(back_populates="model_versions")
    training_runs: Mapped[list["TrainingRun"]] = relationship(
        back_populates="model_version"
    )
    eval_runs: Mapped[list["EvalRun"]] = relationship(back_populates="model_version")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), default="running")
    hyperparams: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    model_version: Mapped[ModelVersion] = relationship(back_populates="training_runs")


class EvalSuite(Base):
    __tablename__ = "eval_suites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    cases: Mapped[list["EvalCase"]] = relationship(back_populates="suite")


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("eval_suites.id"), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    expectations: Mapped[dict] = mapped_column(JSON, default=dict)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    suite: Mapped[EvalSuite] = relationship(back_populates="cases")
    results: Mapped[list["EvalResult"]] = relationship(back_populates="case")


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    suite_id: Mapped[int] = mapped_column(ForeignKey("eval_suites.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    summary_metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    model_version: Mapped[ModelVersion] = relationship(back_populates="eval_runs")
    suite: Mapped[EvalSuite] = relationship()
    results: Mapped[list["EvalResult"]] = relationship(back_populates="run")


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id"), nullable=False)
    case_id: Mapped[int] = mapped_column(ForeignKey("eval_cases.id"), nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped[EvalRun] = relationship(back_populates="results")
    case: Mapped[EvalCase] = relationship(back_populates="results")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id"), nullable=True
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    training_run: Mapped[Optional[TrainingRun]] = relationship()
    model_version: Mapped[Optional[ModelVersion]] = relationship()


# NOTE: Old screenplay tables (ScriptProject, ScriptScene, ScriptRevision, SceneLink,
# ScriptSceneEmbedding) have been removed. Use the new schema in db/screenplay_schema.py:
# - ScreenplayProject, ScreenplayScene, ScreenplayCharacter, SceneElement,
#   DialogueLine, SceneEmbedding, SceneRelation, SceneRevision, ExportConfig
