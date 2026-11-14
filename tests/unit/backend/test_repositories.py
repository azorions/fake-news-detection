import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from fnd_api.config import Settings
from fnd_api.db.repositories import FeedbackRepository, ScanNotFoundError, ScanRepository
from fnd_api.db.session import init_db, session_factory
from fnd_api.schemas import FeedbackCreate, Verdict
from tests.fixtures.responses import SAMPLE_TEXT, make_response

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    db_engine = await init_db(Settings(var_dir=tmp_path / "nested" / "var"))
    yield db_engine
    await db_engine.dispose()


@pytest.fixture
def scans(engine: AsyncEngine) -> ScanRepository:
    return ScanRepository(session_factory(engine))


@pytest.fixture
def feedback(engine: AsyncEngine) -> FeedbackRepository:
    return FeedbackRepository(session_factory(engine))


async def _add(scans: ScanRepository, **overrides: object) -> str:
    response = make_response(**overrides)
    await scans.add(
        response, input_hash="h" * 64, depth="standard", external_used=False, status="complete"
    )
    return response.scan_id


# ---------------------------------------------------------------- schema


async def test_migrations_create_the_tables_and_indexes(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        rows = await connection.execute(
            text("SELECT type, name, tbl_name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
        )
        found = {(kind, name, table) for kind, name, table in rows}
        version = (
            await connection.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar()
    tables = {name for kind, name, _ in found if kind == "table"}
    assert {"scans", "feedback", "source_reputation"} <= tables
    scan_indexes = {name for kind, name, table in found if kind == "index" and table == "scans"}
    assert scan_indexes == {
        "ix_scans_created",
        "ix_scans_hash",
        "ix_scans_domain",
        "ix_scans_verdict",
    }
    assert ("index", "ix_feedback_scan", "feedback") in found
    assert version == "0001"


async def test_connections_use_wal_and_enforce_foreign_keys(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        assert (await connection.execute(text("PRAGMA journal_mode"))).scalar() == "wal"
        assert (await connection.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_init_db_is_safe_to_run_again(tmp_path: Path) -> None:
    settings = Settings(var_dir=tmp_path)
    first = await init_db(settings)
    await ScanRepository(session_factory(first)).add(
        make_response(scan_id="KEEP"), input_hash="h", depth="quick", external_used=False,
        status="complete",
    )  # fmt: skip
    await first.dispose()
    second = await init_db(settings)
    assert await ScanRepository(session_factory(second)).get("KEEP") is not None
    await second.dispose()


async def test_check_constraints_reject_unknown_values(engine: AsyncEngine) -> None:
    from sqlalchemy.exc import IntegrityError

    insert = text(
        "INSERT INTO source_reputation (domain, category, provenance, updated_at) "
        "VALUES ('example.com', :category, '[]', '2026-01-01T00:00:00Z')"
    )
    async with engine.begin() as connection:
        await connection.execute(insert, {"category": "reliable"})
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            await connection.execute(insert, {"category": "wonderful"})


# ---------------------------------------------------------------- scans


async def test_add_then_get_round_trips_the_response(scans: ScanRepository) -> None:
    response = make_response(created_at=T0)
    await scans.add(
        response, input_hash="a" * 64, depth="standard", external_used=False, status="complete"
    )
    assert await scans.get(response.scan_id) == response
    assert await scans.get("missing") is None


async def test_stored_columns_mirror_the_response(
    scans: ScanRepository, engine: AsyncEngine
) -> None:
    scan_id = await _add(
        scans,
        created_at=T0,
        article={"input_type": "url", "url": "https://example.com/a?utm=1"},
        source={"domain": "example.com"},
        metadata={"title": "A headline"},
    )
    async with engine.connect() as connection:
        row = (
            (await connection.execute(text("SELECT * FROM scans WHERE id = :id"), {"id": scan_id}))
            .mappings()
            .one()
        )
    assert row["created_at"] == "2026-11-01T12:00:00.000000Z"
    assert row["input_type"] == "url"
    assert row["domain"] == "example.com"
    assert row["title"] == "A headline"
    assert row["input_text"] == SAMPLE_TEXT
    assert row["verdict"] == "high_risk"
    assert row["classifier_prob"] == pytest.approx(0.91)
    assert row["external_used"] == 0
    assert row["model_versions"] == '{"baseline": "v-test"}'
    assert row["schema_version"] == "1.0"


async def test_input_text_is_not_stored_when_disabled(engine: AsyncEngine) -> None:
    private = ScanRepository(session_factory(engine), store_input_text=False)
    scan_id = await _add(private)
    stored = await private.get(scan_id)
    assert stored is not None
    assert stored.article.text == ""
    assert stored.highlights == []
    assert stored.attributions == []
    assert stored.verdict == Verdict.HIGH_RISK
    async with engine.connect() as connection:
        dump = (await connection.execute(text("SELECT input_text, result_json FROM scans"))).one()
    assert dump[0] is None
    assert "vaccines" not in dump[1]
    assert "shocking" not in dump[1].lower()


async def test_list_is_newest_first_and_the_cursor_continues(scans: ScanRepository) -> None:
    ids = [await _add(scans, created_at=T0 + timedelta(minutes=i)) for i in range(5)]
    newest_first = ids[::-1]

    first, cursor = await scans.list(limit=2)
    assert [item.scan_id for item in first] == newest_first[:2]
    assert cursor is not None
    second, cursor = await scans.list(cursor=cursor, limit=2)
    assert [item.scan_id for item in second] == newest_first[2:4]
    third, cursor = await scans.list(cursor=cursor, limit=2)
    assert [item.scan_id for item in third] == newest_first[4:]
    assert cursor is None
    assert first[0].created_at == T0 + timedelta(minutes=4)


async def test_list_orders_scans_created_in_the_same_instant(scans: ScanRepository) -> None:
    for scan_id in ("01A", "01C", "01B"):
        await _add(scans, scan_id=scan_id, created_at=T0)
    page, cursor = await scans.list(limit=2)
    rest, end = await scans.list(cursor=cursor, limit=2)
    assert [item.scan_id for item in [*page, *rest]] == ["01C", "01B", "01A"]
    assert end is None


async def test_list_filters_by_verdict_domain_and_query(scans: ScanRepository) -> None:
    risky = await _add(scans, created_at=T0, metadata={"title": "Miracle cure exposed"})
    calm = await _add(
        scans,
        created_at=T0 + timedelta(minutes=1),
        verdict="reliable",
        source={"domain": "example.com"},
        metadata={"title": "Budget approved 100%"},
    )
    assert [s.scan_id for s in (await scans.list(verdict=Verdict.RELIABLE))[0]] == [calm]
    assert [s.scan_id for s in (await scans.list(domain="example.com"))[0]] == [calm]
    assert [s.scan_id for s in (await scans.list(q="miracle"))[0]] == [risky]
    assert [s.scan_id for s in (await scans.list(q="100%"))[0]] == [calm]
    assert (await scans.list(q="%"))[0][0].scan_id == calm  # a literal percent sign, not a wildcard
    assert len((await scans.list(q="%"))[0]) == 1
    assert (await scans.list(q="nothing matches this"))[0] == []


async def test_list_rejects_a_malformed_cursor(scans: ScanRepository) -> None:
    with pytest.raises(ValueError, match="cursor"):
        await scans.list(cursor="not-a-cursor")


async def test_concurrent_writes_all_succeed(scans: ScanRepository) -> None:
    ids = await asyncio.gather(*(_add(scans) for _ in range(12)))
    listed, _ = await scans.list(limit=50)
    assert {item.scan_id for item in listed} == set(ids)


# ---------------------------------------------------------------- feedback


async def test_feedback_is_stored_and_counted(
    scans: ScanRepository, feedback: FeedbackRepository
) -> None:
    scan_id = await _add(scans)
    created = await feedback.add(
        scan_id,
        FeedbackCreate(
            rating="disagree", corrected_verdict=Verdict.RELIABLE, comment="It is satire"
        ),
    )
    await feedback.add(
        scan_id, FeedbackCreate(rating="agree", target_type="highlight", target_id="s1")
    )
    assert created.scan_id == scan_id
    assert created.corrected_verdict == Verdict.RELIABLE
    assert created.comment == "It is satire"
    assert created.created_at.tzinfo is not None
    assert await feedback.count(scan_id) == 2
    assert (await scans.list())[0][0].feedback_count == 2


async def test_feedback_for_an_unknown_scan_is_refused(feedback: FeedbackRepository) -> None:
    with pytest.raises(ScanNotFoundError):
        await feedback.add("missing", FeedbackCreate(rating="agree"))


async def test_deleting_a_scan_removes_its_feedback(
    scans: ScanRepository, feedback: FeedbackRepository
) -> None:
    scan_id = await _add(scans)
    other = await _add(scans)
    await feedback.add(scan_id, FeedbackCreate(rating="agree"))
    await feedback.add(other, FeedbackCreate(rating="agree"))

    assert await scans.delete(scan_id) is True
    assert await scans.delete(scan_id) is False
    assert await scans.get(scan_id) is None
    assert await feedback.count(scan_id) == 0
    assert await feedback.count(other) == 1
