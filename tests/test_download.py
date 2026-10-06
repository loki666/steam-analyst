import pytest

from steam_game_analyst.data import download

URL = "https://data.mendeley.com/public-files/datasets/x/files/y/file_downloaded"


def test_mendeley_download_url_reads_snake_case_api_shape():
    entry = {"filename": "Game Reviews.zip", "content_details": {"download_url": URL}}
    assert download.mendeley_download_url(entry) == URL


def test_mendeley_download_url_reads_legacy_camel_case_shape():
    entry = {"filename": "games.json", "contentDetails": {"downloadUrl": URL}}
    assert download.mendeley_download_url(entry) == URL


def test_mendeley_download_url_names_the_file_when_url_missing():
    with pytest.raises(ValueError, match="games.json"):
        download.mendeley_download_url({"filename": "games.json", "content_details": {}})
