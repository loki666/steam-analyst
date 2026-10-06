"""Raw source downloads. Each function is idempotent: skips files that already exist.

Sources
-------
catalog : FronkonGames/steam-games-dataset on Hugging Face (MIT)         -> data/raw/catalog/
reviews : Mendeley Data jxy85cr3th v2 (31M reviews, 2020-2024)          -> data/raw/reviews/
news    : Steam Web API ISteamNews/GetNewsForApp for the subset appids  -> data/raw/news/<appid>.json
wiki    : see data/wiki.py (Fandom MediaWiki API per game)
"""

from __future__ import annotations

import json
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path

import httpx
from rich.console import Console
from tqdm import tqdm

from steam_game_analyst.config import NewsConfig

console = Console()

HF_API = "https://huggingface.co/api/datasets/{repo}/tree/main"
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
MENDELEY_FILES = (
    "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={version}"
)
STEAM_NEWS = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"


def _stream_download(url: str, dest: Path, client: httpx.Client, desc: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        console.print(f"[dim]skip (exists)[/] {dest}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
        ):
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


# ----------------------------------------------------------------------------- catalog
def download_catalog(raw_dir: Path, hf_repo: str, hf_file: str | None = None) -> Path:
    """Download the catalog file from the HF dataset repo. Prefers parquet, then csv."""
    out_dir = raw_dir / "catalog"
    with httpx.Client(timeout=120) as client:
        files = (
            client.get(HF_API.format(repo=hf_repo), follow_redirects=True).raise_for_status().json()
        )
        names = [f["path"] for f in files if f.get("type") == "file"]
        pick = None
        if hf_file and hf_file in names:
            pick = hf_file
        else:
            for ext in (".parquet", ".csv", ".json"):
                cands = [n for n in names if n.endswith(ext)]
                if cands:
                    pick = sorted(cands)[0]
                    break
        if not pick:
            raise RuntimeError(f"No data file found in {hf_repo}: {names}")
        return _stream_download(
            HF_FILE.format(repo=hf_repo, path=pick), out_dir / Path(pick).name, client
        )


# ----------------------------------------------------------------------------- reviews
def mendeley_download_url(entry: dict) -> str:
    """Download URL of one Mendeley file entry (public API is snake_case; older docs camelCase)."""
    details = entry.get("content_details") or entry.get("contentDetails") or {}
    url = details.get("download_url") or details.get("downloadUrl")
    if not url:
        raise ValueError(f"no download url for Mendeley file {entry.get('filename')!r}")
    return url


def download_reviews(
    raw_dir: Path, mendeley_id: str, version: int = 2, extract: bool = True
) -> Path:
    """Download all files of the Mendeley dataset (one or more zips) and extract them."""
    out_dir = raw_dir / "reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=None) as client:
        meta = (
            client.get(MENDELEY_FILES.format(id=mendeley_id, version=version))
            .raise_for_status()
            .json()
        )
        for f in meta:
            url = mendeley_download_url(f)
            name = f["filename"]
            dest = _stream_download(url, out_dir / name, client, desc=name)
            if extract and name.lower().endswith(".zip"):
                marker = out_dir / f".{name}.extracted"
                if not marker.exists():
                    console.print(f"extracting {name} ...")
                    with zipfile.ZipFile(dest) as z:
                        z.extractall(out_dir)
                    marker.touch()
    return out_dir


# ----------------------------------------------------------------------------- news
def download_news(
    raw_dir: Path, appids: Iterable[int], api_key: str | None, cfg: NewsConfig
) -> Path:
    """GetNewsForApp for each appid -> data/raw/news/<appid>.json (one call per app)."""
    out_dir = raw_dir / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    delay = 1.0 / max(cfg.rate_limit_per_sec, 0.1)
    with httpx.Client(timeout=60) as client:
        for appid in tqdm(list(appids), desc="news"):
            dest = out_dir / f"{appid}.json"
            if dest.exists():
                continue
            params: dict[str, str | int] = {
                "appid": appid,
                "count": cfg.count,
                "maxlength": cfg.maxlength,
                "format": "json",
            }
            if cfg.feeds:
                params["feeds"] = cfg.feeds
            if api_key:
                params["key"] = api_key
            for attempt in range(4):
                r = client.get(STEAM_NEWS, params=params)
                if r.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                if r.status_code == 403 and "key" in params:
                    # some apps reject keyed requests; retry anonymously
                    params.pop("key")
                    continue
                break
            if r.status_code != 200:
                console.print(f"[yellow]news {appid}: HTTP {r.status_code}[/]")
                dest.write_text(
                    json.dumps(
                        {"appnews": {"appid": appid, "newsitems": [], "error": r.status_code}}
                    )
                )
            else:
                dest.write_text(r.text)
            time.sleep(delay)
    return out_dir
