"""ORM models. They mirror the DDL in docs/design.md, section 3; timestamps are ISO-8601 UTC text."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, ForeignKey, Index, Integer, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import REAL


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # ULID
    created_at: Mapped[str] = mapped_column(Text)
    input_type: Mapped[str] = mapped_column(Text)
    input_hash: Mapped[str] = mapped_column(Text)  # sha256(canonical URL or normalized text)
    url: Mapped[str | None] = mapped_column(Text)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(Text)  # registrable domain
    title: Mapped[str | None] = mapped_column(Text)
    input_text: Mapped[str | None] = mapped_column(Text)  # NULL when input text is not stored
    word_count: Mapped[int] = mapped_column(Integer)
    verdict: Mapped[str] = mapped_column(Text)
    risk_score: Mapped[float] = mapped_column(REAL)
    confidence: Mapped[float] = mapped_column(REAL)
    classifier_prob: Mapped[float | None] = mapped_column(REAL)
    depth: Mapped[str] = mapped_column(Text)
    external_used: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    status: Mapped[str] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int] = mapped_column(Integer)
    llm_input_tokens: Mapped[int | None] = mapped_column(Integer)
    llm_output_tokens: Mapped[int | None] = mapped_column(Integer)
    model_versions: Mapped[str] = mapped_column(Text)  # JSON
    schema_version: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)  # full AnalysisResponse snapshot

    __table_args__ = (
        CheckConstraint("input_type IN ('url','text','headline')", name="ck_scans_input_type"),
        CheckConstraint(
            "verdict IN ('reliable','potentially_misleading','high_risk','unverified')",
            name="ck_scans_verdict",
        ),
        CheckConstraint("status IN ('complete','partial','failed')", name="ck_scans_status"),
        Index("ix_scans_created", text("created_at DESC")),
        Index("ix_scans_hash", "input_hash"),
        Index("ix_scans_domain", "domain"),
        Index("ix_scans_verdict", "verdict"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    scan_id: Mapped[str] = mapped_column(Text, ForeignKey("scans.id", ondelete="CASCADE"))
    created_at: Mapped[str] = mapped_column(Text)
    rating: Mapped[str] = mapped_column(Text)
    corrected_verdict: Mapped[str | None] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(Text, server_default=text("'verdict'"))
    target_id: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("rating IN ('agree','disagree')", name="ck_feedback_rating"),
        CheckConstraint(
            "target_type IN ('verdict','claim','highlight','source')",
            name="ck_feedback_target_type",
        ),
        Index("ix_feedback_scan", "scan_id"),
    )


class SourceReputation(Base):
    __tablename__ = "source_reputation"

    domain: Mapped[str] = mapped_column(Text, primary_key=True)  # registrable domain, lowercase
    credibility_score: Mapped[float | None] = mapped_column(REAL)  # 0..1, NULL if only categorical
    category: Mapped[str] = mapped_column(Text)
    is_satire: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    provenance: Mapped[str] = mapped_column(Text)  # JSON: [{list, rating, as_of}]
    updated_at: Mapped[str] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "category IN ('reliable','mixed','unreliable','satire','unknown')",
            name="ck_source_reputation_category",
        ),
    )
