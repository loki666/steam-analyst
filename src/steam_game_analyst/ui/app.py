"""Streamlit UI (W5). W1 placeholder: shows pipeline status so `docker compose up` has something to render."""

from __future__ import annotations

import streamlit as st

from steam_game_analyst.config import get_settings

st.set_page_config(page_title="Steam Game Analyst", page_icon="🎮", layout="wide")
st.title("Steam Game Analyst")
s = get_settings()
st.caption("W1 skeleton. The agent UI arrives in W5.")
db = s.duckdb_path
if db.exists():
    import duckdb

    con = duckdb.connect(str(db), read_only=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Games", con.execute("select count(*) from games").fetchone()[0])
    c2.metric("Reviews", con.execute("select count(*) from reviews").fetchone()[0])
    c3.metric("News items", con.execute("select count(*) from news").fetchone()[0])
    st.dataframe(con.execute("select * from game_summary order by positive desc limit 50").df())
    con.close()
else:
    st.info(f"No data yet. Run `make data` (expects {db}).")
