"""Initial schema: scans, feedback, source_reputation.

Revision ID: 0001
Revises:
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scans",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("input_type", sa.Text(), nullable=False),
        sa.Column("input_hash", sa.Text(), nullable=False),
        sa.Column("url", sa.Text()),
        sa.Column("canonical_url", sa.Text()),
        sa.Column("domain", sa.Text()),
        sa.Column("title", sa.Text()),
        sa.Column("input_text", sa.Text()),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("risk_score", sa.REAL(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=False),
        sa.Column("classifier_prob", sa.REAL()),
        sa.Column("depth", sa.Text(), nullable=False),
        sa.Column("external_used", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_code", sa.Text()),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("llm_input_tokens", sa.Integer()),
        sa.Column("llm_output_tokens", sa.Integer()),
        sa.Column("model_versions", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=False),
        sa.CheckConstraint("input_type IN ('url','text','headline')", name="ck_scans_input_type"),
        sa.CheckConstraint(
            "verdict IN ('reliable','potentially_misleading','high_risk','unverified')",
            name="ck_scans_verdict",
        ),
        sa.CheckConstraint("status IN ('complete','partial','failed')", name="ck_scans_status"),
    )
    op.create_index("ix_scans_created", "scans", [sa.text("created_at DESC")])
    op.create_index("ix_scans_hash", "scans", ["input_hash"])
    op.create_index("ix_scans_domain", "scans", ["domain"])
    op.create_index("ix_scans_verdict", "scans", ["verdict"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "scan_id", sa.Text(), sa.ForeignKey("scans.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("rating", sa.Text(), nullable=False),
        sa.Column("corrected_verdict", sa.Text()),
        sa.Column("target_type", sa.Text(), nullable=False, server_default=sa.text("'verdict'")),
        sa.Column("target_id", sa.Text()),
        sa.Column("comment", sa.Text()),
        sa.CheckConstraint("rating IN ('agree','disagree')", name="ck_feedback_rating"),
        sa.CheckConstraint(
            "target_type IN ('verdict','claim','highlight','source')",
            name="ck_feedback_target_type",
        ),
    )
    op.create_index("ix_feedback_scan", "feedback", ["scan_id"])

    op.create_table(
        "source_reputation",
        sa.Column("domain", sa.Text(), primary_key=True),
        sa.Column("credibility_score", sa.REAL()),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("is_satire", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("provenance", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "category IN ('reliable','mixed','unreliable','satire','unknown')",
            name="ck_source_reputation_category",
        ),
    )


def downgrade() -> None:
    op.drop_table("source_reputation")
    op.drop_index("ix_feedback_scan", table_name="feedback")
    op.drop_table("feedback")
    for index in ("ix_scans_verdict", "ix_scans_domain", "ix_scans_hash", "ix_scans_created"):
        op.drop_index(index, table_name="scans")
    op.drop_table("scans")
