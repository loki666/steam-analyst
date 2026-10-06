import pytest

from steam_game_analyst.data import transform

# FronkonGames games.csv ships a header with "Discount" and "DLC count" merged into one
# column, so every field after Price is one column short of its header.
BROKEN_CSV = (
    "AppID,Name,Release date,Price,DiscountDLC count,Developers,Genres,Positive,Negative\n"
    '1245620,ELDEN RING,"Feb 24, 2022",59.99,0,3,FromSoftware,"Action,RPG",981540,120000\n'
)


def test_read_catalog_raw_repairs_merged_discount_dlc_header(tmp_path):
    p = tmp_path / "games.csv"
    p.write_text(BROKEN_CSV)

    df = transform.read_catalog_raw(p)

    row = df.row(0, named=True)
    assert "Discount" in df.columns and "DLC count" in df.columns
    assert row["DLC count"] == 3
    assert row["Developers"] == "FromSoftware"
    assert row["Genres"] == "Action,RPG"
    assert row["Positive"] == 981540
    assert row["Negative"] == 120000


def test_read_catalog_raw_leaves_well_formed_header_alone(tmp_path):
    p = tmp_path / "games.csv"
    p.write_text("AppID,Name,Price\n10,Counter-Strike,9.99\n")

    df = transform.read_catalog_raw(p)

    assert df.columns == ["AppID", "Name", "Price"]
    assert df.row(0, named=True)["Price"] == 9.99


def test_read_catalog_raw_fails_loudly_when_repair_does_not_fit(tmp_path):
    # header still merged, but rows no longer carry the extra field -> layout changed upstream
    p = tmp_path / "games.csv"
    p.write_text("AppID,Name,Price,DiscountDLC count,Genres\n10,CS,9.99,0,Action\n")

    with pytest.raises(ValueError, match="layout changed"):
        transform.read_catalog_raw(p)
