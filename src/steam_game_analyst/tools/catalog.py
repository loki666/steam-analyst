"""query_catalog: read-only SQL over the DuckDB tables, guarded in two layers.

1. Parser check (no binding): exactly one SELECT statement, no table functions, and only
   allow-listed tables (CTE names are fine).
2. Sandboxed connection: the database is attached read-only, file access is limited to
   data/processed/ and the configuration is locked, so anything the parser check misses
   (getenv, COPY, ATTACH, INSTALL, ...) still fails.
Results are capped at MAX_ROWS rows and the query is interrupted after TIMEOUT_S seconds."""

from __future__ import annotations

import difflib
import json
import re
import threading
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

MAX_SQL_CHARS = 4000


_LIST_HINT = (
    " -- tags, genres, categories, developers, publishers and supported_languages are lists "
    "(VARCHAR[]): use list_contains(tags, 'Exact Name'), or "
    "len(list_filter(tags, t -> t ILIKE '%word%')) > 0 for a partial match"
)


_LIST_COLUMNS = ("tags", "genres", "categories")
_RATED_TABLES = ("games", "games_all", "game_summary")
# views over parquet files in sga.duckdb (data/build.py DUCKDB_DDL); the rest are tables in it
PARQUET_TABLES = frozenset({"games", "games_all", "reviews", "news"})
# n_votes and the Wilson 95% lower bound of the positive ratio: ranking by the bound keeps a
# game with 3/3 positive votes below one with 9,500/10,000
_RATING_COLUMNS = """,
    positive + negative AS n_votes,
    CASE WHEN positive + negative > 0 THEN
      (positive_ratio + 1.96^2 / (2 * (positive + negative))
       - 1.96 * sqrt(positive_ratio * (1 - positive_ratio) / (positive + negative)
                     + 1.96^2 / (4 * (positive + negative)^2)))
      / (1 + 1.96^2 / (positive + negative))
    END AS positive_ratio_lower95"""
_LIST_LITERAL = re.compile(
    r"list_(?:contains|has|has_any|has_all)\s*\(\s*\w+\s*,\s*(?:\[\s*)?'([^']{2,60})'", re.I
)


def _norm(s: str) -> str:
    return "".join(ch for ch in s.casefold() if ch.isalnum())


class SqlGuardError(ValueError):
    """The query was refused or failed; the message is safe to show the model."""


MAX_CELL_CHARS = 500


def _jsonable(v: Any) -> Any:
    if isinstance(v, str) and len(v) > MAX_CELL_CHARS:
        return v[:MAX_CELL_CHARS] + "…"
    if isinstance(v, datetime | date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    return v


def _walk(node: Any, tables: set[str], ctes: set[str], funcs: list[str]) -> None:
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "BASE_TABLE":
            name = node.get("table_name", "")
            catalog, schema = node.get("catalog_name", ""), node.get("schema_name", "")
            # only a bare name can refer to a CTE or an allowed view; qualified names are refused
            tables.add(name if not (catalog or schema) else f"{catalog}.{schema}.{name}")
        elif kind == "TABLE_FUNCTION":
            fn = node.get("function", {})
            funcs.append(fn.get("function_name", "?") if isinstance(fn, dict) else "?")
        for entry in (
            node.get("cte_map", {}).get("map", []) if isinstance(node.get("cte_map"), dict) else []
        ):
            ctes.add(entry.get("key", ""))
        for v in node.values():
            _walk(v, tables, ctes, funcs)
    elif isinstance(node, list):
        for v in node:
            _walk(v, tables, ctes, funcs)


def check_sql(sql: str, allowed: tuple[str, ...] = ()) -> str:
    """Parser-level guard; returns the query without trailing semicolons."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise SqlGuardError("empty query")
    if len(sql) > MAX_SQL_CHARS:
        raise SqlGuardError(f"query longer than {MAX_SQL_CHARS} characters")
    con = duckdb.connect(":memory:")
    try:
        raw = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()
    finally:
        con.close()
    tree = json.loads(raw[0]) if raw else {"error": True}
    if tree.get("error"):
        msg = tree.get("error_message", "")
        raise SqlGuardError(f"only a single SELECT statement is allowed ({msg})".strip())
    if len(tree.get("statements", [])) != 1:
        raise SqlGuardError("only a single SELECT statement is allowed")
    tables: set[str] = set()
    ctes: set[str] = set()
    funcs: list[str] = []
    _walk(tree["statements"], tables, ctes, funcs)
    if funcs:
        raise SqlGuardError(f"table functions are not allowed: {sorted(set(funcs))}")
    unknown = sorted(t for t in tables - ctes if t not in allowed)
    if allowed and (ctes & set(allowed)):
        raise SqlGuardError(f"a CTE may not reuse a table name: {sorted(ctes & set(allowed))}")
    if allowed and unknown:
        raise SqlGuardError(f"unknown or forbidden tables {unknown}; allowed: {list(allowed)}")
    return sql


class CatalogDB:
    TABLES: tuple[str, ...] = (
        "games",
        "games_all",
        "game_summary",
        "review_monthly",
        "reviews",
        "news",
    )
    MAX_ROWS = 200
    TIMEOUT_S = 20.0

    def __init__(self, duckdb_path: Path) -> None:
        if not duckdb_path.exists():
            raise FileNotFoundError(f"{duckdb_path} missing; run `sga data build duckdb`")
        data_dir = duckdb_path.parent.resolve()
        self._con = duckdb.connect(":memory:")
        self._con.execute(f"ATTACH '{duckdb_path.resolve().as_posix()}' AS sga (READ_ONLY)")
        # queries run in the in-memory catalog, where only the allow-listed relations exist.
        # Parquet-backed tables are read from this file's own directory: the views stored in
        # sga.duckdb hold the absolute paths of the machine that built it.
        for t in self.TABLES:
            extra = _RATING_COLUMNS if t in _RATED_TABLES else ""
            source = (
                f"read_parquet('{(data_dir / f'{t}.parquet').as_posix()}')"
                if t in PARQUET_TABLES
                else f"sga.main.{t}"
            )
            self._con.execute(f"CREATE VIEW memory.main.{t} AS SELECT *{extra} FROM {source}")
        self._con.execute("USE memory")
        self._con.execute("SET memory_limit = '1GB'")
        self._con.execute(f"SET allowed_directories = ['{data_dir.as_posix()}/']")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")
        self._lock = threading.Lock()
        self._vocab: dict[str, list[str]] | None = None

    def vocabulary(self) -> dict[str, list[str]]:
        """Distinct tag / genre / category names in the full catalog."""
        if self._vocab is None:
            self._vocab = {
                col: [
                    r[0]
                    for r in self.fetch(
                        f"SELECT DISTINCT x FROM (SELECT unnest({col}) AS x FROM games_all) "
                        "WHERE x IS NOT NULL"
                    )
                ]
                for col in _LIST_COLUMNS
            }
        return self._vocab

    def suggest(self, sql: str, k: int = 8) -> dict[str, list[str]]:
        """Existing list names close to the literals the query looked for."""
        wanted = _LIST_LITERAL.findall(sql)
        out: dict[str, list[str]] = {}
        for col, names in self.vocabulary().items():
            by_norm = {_norm(n): n for n in names}
            hits: list[str] = []
            for w in wanted:
                nw = _norm(w)
                hits += [
                    n
                    for key, n in by_norm.items()
                    if nw and (nw in key or key in nw) and len(key) >= 3
                ]
                hits += [by_norm[m] for m in difflib.get_close_matches(nw, list(by_norm), 5, 0.8)]
            if hits:
                out[col] = list(dict.fromkeys(sorted(hits, key=len)))[:k]
        return out

    def close(self) -> None:
        self._con.close()

    def query(self, sql: str, limit: int = 50) -> dict[str, Any]:
        sql = check_sql(sql, self.TABLES)
        limit = max(1, min(int(limit), self.MAX_ROWS))
        with self._lock:
            timer = threading.Timer(self.TIMEOUT_S, self._con.interrupt)
            timer.start()
            try:
                cur = self._con.execute(self._bounded(sql, limit + 1))
                rows = cur.fetchall()
                columns = [d[0] for d in cur.description or []]
                total = (
                    self._con.execute(f"SELECT count(*) FROM ({sql}) AS q").fetchall()[0][0]
                    if len(rows) > limit
                    else None
                )
            except duckdb.InterruptException as e:
                raise SqlGuardError(f"query took longer than {self.TIMEOUT_S:.0f}s") from e
            except duckdb.Error as e:
                msg = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                if "[]" in msg:  # a list column used like a string
                    msg += _LIST_HINT
                raise SqlGuardError(msg) from e
            finally:
                timer.cancel()
        # counts first: the registry cuts long outputs by characters, from the end
        out: dict[str, Any] = {
            "columns": columns,
            "row_count": min(len(rows), limit),
            "truncated": total is not None,
            **({"total_rows": total} if total is not None else {}),
            "rows": [[_jsonable(v) for v in r] for r in rows[:limit]],
        }
        if not rows:
            out["hint"] = (
                "0 rows. Check the table (games = 300-game review subset, games_all = full "
                "catalog), exact list names (tags, genres and categories are different lists; "
                "for a family of tags use len(list_filter(tags, t -> t ILIKE '%word%')) > 0) "
                "and whether a filter is too strict."
            )
            if suggestions := self.suggest(sql):
                looked_for = ", ".join(repr(w) for w in _LIST_LITERAL.findall(sql))
                existing = "; ".join(f"{col}: {names}" for col, names in suggestions.items())
                out = {
                    "no_match": f"nothing matched {looked_for}. Names that exist: {existing}. "
                    "Retry with these names in the right list column.",
                    **out,
                    "did_you_mean": suggestions,
                }
        return out

    def _bounded(self, sql: str, n: int) -> str:
        """Wrap the checked query so DuckDB itself caps rows and text cells: a huge value never
        reaches Python."""
        cols = self._con.execute(f"DESCRIBE ({sql})").fetchall()
        exprs = [
            f'left(q."{c[0]}", {MAX_CELL_CHARS + 1}) AS "{c[0]}"'
            if c[1] == "VARCHAR"
            else f'q."{c[0]}"'
            for c in cols
        ]
        return f"SELECT {', '.join(exprs) or '*'} FROM ({sql}) AS q LIMIT {int(n)}"

    def fetch(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        """Trusted, parameterised internal query for the other tools (no guard, same sandbox)."""
        with self._lock:
            return self._con.execute(sql, params or []).fetchall()

    def schema(self) -> dict[str, list[str]]:
        """table -> ["column TYPE", ...] for the allowed tables."""
        out: dict[str, list[str]] = {}
        with self._lock:
            for t in self.TABLES:
                cols = self._con.execute(f"DESCRIBE {t}").fetchall()
                out[t] = [f"{c[0]} {c[1]}" for c in cols]
        return out
