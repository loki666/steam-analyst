"""query_catalog: guarded, read-only SQL over DuckDB."""

import pytest

from steam_game_analyst.tools.catalog import CatalogDB, SqlGuardError, check_sql


@pytest.fixture
def db(built):
    d = CatalogDB(built.duckdb_path)
    yield d
    d.close()


def test_select_over_allowed_tables_returns_rows(db):
    res = db.query("SELECT name, price_usd FROM games ORDER BY price_usd DESC", limit=2)
    assert res["columns"] == ["name", "price_usd"]
    assert len(res["rows"]) == 2 and res["truncated"] is True
    assert res["rows"][0][0] in {"ELDEN RING", "Cyberpunk 2077"}


def test_ctes_joins_and_subqueries_are_allowed(db):
    sql = (
        "WITH g AS (SELECT appid, name FROM games) "
        "SELECT g.name, count(*) n FROM g JOIN reviews r USING (appid) "
        "WHERE appid IN (SELECT appid FROM game_summary) GROUP BY 1 ORDER BY 2 DESC"
    )
    assert db.query(sql)["rows"]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM games",
        "DROP TABLE review_monthly",
        "INSERT INTO games SELECT * FROM games",
        "CREATE TABLE x AS SELECT 1",
        "SELECT 1; SELECT 2",
        "ATTACH '/tmp/x.db'",
        "COPY games TO '/tmp/games.csv'",
        "PRAGMA database_list",
        "SET threads = 1",
        "INSTALL httpfs",
    ],
)
def test_non_select_statements_are_rejected(db, sql):
    with pytest.raises(SqlGuardError):
        db.query(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_parquet('data/processed/games.parquet')",
        "SELECT * FROM duckdb_settings()",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM other_db.main.games",
    ],
)
def test_table_functions_and_unlisted_tables_are_rejected(db, sql):
    with pytest.raises(SqlGuardError):
        db.query(sql)


def test_file_access_is_blocked_even_past_the_parser(db):
    # scalar route to the file system: the sandboxed connection refuses it
    with pytest.raises(SqlGuardError):
        db.query("SELECT getenv('HOME')")


def test_check_sql_strips_trailing_semicolon_and_rejects_long_queries():
    assert check_sql("SELECT 1;").strip() == "SELECT 1"
    with pytest.raises(SqlGuardError):
        check_sql("SELECT " + "1+" * 3000 + "1")


def test_row_limit_is_capped(db):
    res = db.query("SELECT r.appid FROM reviews r, games g", limit=10_000)  # 480 rows
    assert len(res["rows"]) == CatalogDB.MAX_ROWS


def test_schema_lists_only_allowed_tables(db):
    schema = db.schema()
    assert set(schema) == set(CatalogDB.TABLES)
    assert "price_usd DOUBLE" in schema["games_all"]


def test_like_on_a_list_column_explains_list_functions(db):
    with pytest.raises(SqlGuardError, match="list_contains"):
        db.query("SELECT name FROM games WHERE tags LIKE '%RPG%'")


def test_empty_result_carries_a_hint(db):
    res = db.query("SELECT name FROM games WHERE name = 'nothing like this'")
    assert res["rows"] == [] and "games_all" in res["hint"]


def test_empty_result_suggests_existing_list_names(db):
    # fixture tags include "Rogue-like" and "Action Roguelike"
    res = db.query("SELECT name FROM games_all WHERE list_contains(tags, 'roguelike')")
    assert res["rows"] == []
    assert set(res["did_you_mean"]["tags"]) >= {"Rogue-like", "Action Roguelike"}
    res = db.query("SELECT name FROM games_all WHERE list_contains(categories, 'Survival')")
    assert "Survival" in res["did_you_mean"]["tags"]  # the name lives in another list


def test_cte_name_cannot_unlock_a_table_outside_the_allow_list(db):
    # review_volume exists in sga.duckdb but is not allow-listed
    for sql in (
        "SELECT * FROM review_volume, (WITH review_volume AS (SELECT 1) SELECT 1) z",
        "WITH review_volume AS (SELECT 1) SELECT * FROM sga.main.review_volume",
    ):
        with pytest.raises(SqlGuardError):
            db.query(sql)


def test_huge_values_are_bounded(db):
    res = db.query("SELECT repeat('a', 200000000) AS s")  # 200 MB string, cut inside DuckDB
    assert len(res["rows"][0][0]) <= 502


def test_rating_views_expose_votes_and_a_wilson_lower_bound(db):
    res = db.query(
        "SELECT name, n_votes, positive_ratio, positive_ratio_lower95 FROM games_all "
        "WHERE n_votes > 0 ORDER BY positive_ratio_lower95 DESC"
    )
    rows = {r[0]: r for r in res["rows"]}
    # Tiny Shooter: 50 / 110 votes; the bound sits below the raw ratio
    tiny = rows["Tiny Shooter"]
    assert tiny[1] == 110 and tiny[3] < tiny[2]
    # big, well-reviewed games outrank it
    assert res["rows"][-1][0] == "Tiny Shooter"


def test_data_directory_can_move_after_the_build(built, tmp_path):
    """sga.duckdb built on one machine opens elsewhere (host -> container, dev box -> VM)."""
    import shutil

    moved = tmp_path / "elsewhere" / "processed"
    shutil.copytree(built.processed_dir, moved)
    shutil.rmtree(built.processed_dir)  # the paths the build saw no longer exist
    db = CatalogDB(moved / "sga.duckdb")
    try:
        for table in CatalogDB.TABLES:
            assert db.fetch(f"SELECT count(*) FROM {table}")[0][0] >= 0
        assert db.query("SELECT count(*) AS n FROM games")["rows"][0][0] > 0
    finally:
        db.close()
