#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
import sqlite3
import subprocess
import threading
import datetime as dt
import importlib
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Form
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
CFG_DIR = DATA_DIR / "Config"
LOG_DIR = DATA_DIR / "Logs"
DB_PATH = DATA_DIR / "tv_ui.db"
APP_CFG = CFG_DIR / "app_config.json"
TV_CFG = ROOT / "config.json"
TV_INDEX = ROOT / "tv_index.json"
TV_QUEUE = ROOT / "tv_queue.json"
TV_MANIFEST = ROOT / "tv_manifest.csv"
RUN_LOG = LOG_DIR / "tv_run.log"
RUN_STATE = LOG_DIR / "tv_run_state.json"
CLZ_CSV = ROOT / "CLZ_export.csv"

DEFAULT_CFG = {
    "rips_staging_root": r"C:\\MediaPipeline\\Ripping_Staging\\1_Raw_Dumps\\TV",
    "rips_complete_root": r"C:\\MediaPipeline\\_QUEUE\\TV",
    "min_clip_seconds": 360,
    "clz_export_path": "",
}

app = FastAPI(title="Just-TV UI", version="0.1.0")
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

RUN_LOCK = threading.Lock()
RUN_PROC: Optional[subprocess.Popen] = None
RUN_STATE_MEM: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "current_step": "idle",
    "current_item": None,
    "last_update": None,
    "returncode": None,
    "return_code": None,
    "error": None,
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def normalize_str(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def parse_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    s = normalize_str(val)
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        pass
    try:
        return int(float(s))
    except Exception:
        pass
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return default
    return default


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_app_cfg() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    raw = read_json(APP_CFG, default={})
    if isinstance(raw, dict):
        cfg.update(raw)
    cfg["min_clip_seconds"] = parse_int(cfg.get("min_clip_seconds"), 360) or 360
    cfg["clz_export_path"] = normalize_str(cfg.get("clz_export_path"))
    return cfg


def validate_cfg(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for key in ("rips_staging_root", "rips_complete_root"):
        if not str(cfg.get(key) or "").strip():
            errors.append(f"{key} is required")
        else:
            p = Path(str(cfg[key]))
            if not p.exists():
                warnings.append(f"{key} does not exist: {p}")
    try:
        v = int(cfg.get("min_clip_seconds") or 0)
        if v <= 0:
            errors.append("min_clip_seconds must be > 0")
    except Exception:
        errors.append("min_clip_seconds must be an integer")
    clz_path = normalize_str(cfg.get("clz_export_path"))
    if clz_path:
        p = Path(clz_path)
        if not p.exists():
            warnings.append(f"clz_export_path does not exist: {p}")
    return {"errors": errors, "warnings": warnings}


def save_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "rips_staging_root": str(cfg.get("rips_staging_root") or "").strip(),
        "rips_complete_root": str(cfg.get("rips_complete_root") or "").strip(),
        "min_clip_seconds": parse_int(cfg.get("min_clip_seconds"), 360) or 360,
        "clz_export_path": normalize_str(cfg.get("clz_export_path")),
    }
    write_json(APP_CFG, out)
    sync_engine_cfg(out)
    return out


def sync_engine_cfg(cfg: Dict[str, Any]) -> None:
    tv_cfg = read_json(TV_CFG, default={})
    if not isinstance(tv_cfg, dict):
        tv_cfg = {}
    tv_cfg["rips_staging_root"] = cfg["rips_staging_root"]
    tv_cfg["final_tv_root"] = cfg["rips_complete_root"]
    tv_cfg["min_clip_seconds"] = int(cfg["min_clip_seconds"])
    write_json(TV_CFG, tv_cfg)


def get_conn() -> sqlite3.Connection:
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS clz_item(
              clz_index TEXT PRIMARY KEY,
              title TEXT NOT NULL,
              barcode TEXT,
              box_set TEXT,
              parent_index TEXT,
              is_tv_series INTEGER DEFAULT 0,
              nr_discs INTEGER DEFAULT 1,
              format TEXT,
              imdb_url TEXT,
              notes TEXT,
              raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS mapped_item(
              queue_key TEXT PRIMARY KEY,
              include_flag INTEGER NOT NULL DEFAULT 1,
              priority INTEGER NOT NULL DEFAULT 0,
              updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS package_link(
              clz_index TEXT PRIMARY KEY,
              tvmaze_show_id INTEGER,
              show_name TEXT,
              imdb_id TEXT,
              selected_season INTEGER,
              updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS episode_cache(
              tvmaze_show_id INTEGER NOT NULL,
              season INTEGER NOT NULL,
              episode_number INTEGER NOT NULL,
              episode_code TEXT NOT NULL,
              episode_title TEXT,
              tvmaze_episode_id INTEGER,
              runtime_minutes INTEGER,
              payload_json TEXT,
              updated_at TEXT,
              PRIMARY KEY(tvmaze_show_id, season, episode_number)
            );
            CREATE TABLE IF NOT EXISTS disc(
              disc_uid TEXT PRIMARY KEY,
              clz_index TEXT NOT NULL,
              disc_number INTEGER NOT NULL,
              label TEXT,
              updated_at TEXT,
              UNIQUE(clz_index, disc_number)
            );
            CREATE TABLE IF NOT EXISTS disc_episode_map(
              disc_uid TEXT NOT NULL,
              episode_code TEXT NOT NULL,
              tvmaze_episode_id INTEGER,
              season INTEGER,
              episode_number INTEGER,
              episode_title TEXT,
              runtime_minutes INTEGER,
              updated_at TEXT,
              PRIMARY KEY(disc_uid, episode_code)
            );
            CREATE TABLE IF NOT EXISTS queue_draft(
              queue_key TEXT PRIMARY KEY,
              position INTEGER NOT NULL,
              included INTEGER NOT NULL DEFAULT 1,
              priority INTEGER NOT NULL DEFAULT 0,
              added_at TEXT
            );
            """
        )


def nav_ctx(current: str, package_clz_index: Optional[str] = None) -> Dict[str, Any]:
    return {
        "nav_current": current,
        "nav_package_clz_index": package_clz_index,
    }


def workflow_ctx() -> Dict[str, Any]:
    with get_conn() as conn:
        clz_loaded = bool(conn.execute("SELECT 1 FROM clz_item LIMIT 1").fetchone())
        mapping_complete = bool(conn.execute("SELECT 1 FROM disc_episode_map LIMIT 1").fetchone())
    queue_built = TV_QUEUE.exists()
    cfg = load_app_cfg()
    cfg_valid = len(validate_cfg(cfg)["errors"]) == 0
    ready_to_run = queue_built and cfg_valid
    if not clz_loaded:
        next_step = "Import your CLZ export to load packages."
    elif not mapping_complete:
        next_step = "Click a title to map discs to episodes."
    elif not queue_built:
        next_step = "Build the rip queue."
    elif not ready_to_run:
        next_step = "Fix config errors, then start the rip run."
    else:
        next_step = "Start the rip run."
    return {
        "wf": {
            "clz_loaded": clz_loaded,
            "mapping_complete": mapping_complete,
            "queue_built": queue_built,
            "ready_to_run": ready_to_run,
            "next_step": next_step,
        }
    }


def make_disc_uid(clz_index: str, disc_number: int) -> str:
    return f"CLZ{normalize_str(clz_index)}-D{int(disc_number):02d}"


def make_queue_key(show_name: str, season: int, disc_number: int) -> str:
    return f"{normalize_str(show_name)}||S{int(season):02d}||D{int(disc_number):02d}"


def parse_episode_code_token(token: str) -> Optional[Tuple[int, int]]:
    m = re.fullmatch(r"s?(\d{1,3})e(\d{1,3})", normalize_str(token).lower())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_episode_codes(text: str) -> List[str]:
    out: List[str] = []
    raw = normalize_str(text)
    if not raw:
        return out
    for part in re.split(r"[,\s]+", raw):
        t = normalize_str(part)
        if not t:
            continue
        if "-" in t:
            left, right = t.split("-", 1)
            a = parse_episode_code_token(left)
            b = parse_episode_code_token(right)
            if not a or not b or a[0] != b[0]:
                continue
            start_ep = min(a[1], b[1])
            end_ep = max(a[1], b[1])
            for ep in range(start_ep, end_ep + 1):
                out.append(f"S{a[0]:02d}E{ep:02d}")
        else:
            x = parse_episode_code_token(t)
            if x:
                out.append(f"S{x[0]:02d}E{x[1]:02d}")
    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def summarize_episode_codes(codes: List[str]) -> str:
    parsed: List[Tuple[int, int]] = []
    for code in codes:
        p = parse_episode_code_token(code)
        if p:
            parsed.append(p)
    if not parsed:
        return "none"
    parsed = sorted(set(parsed), key=lambda x: (x[0], x[1]))
    out: List[str] = []
    i = 0
    while i < len(parsed):
        season, start = parsed[i]
        end = start
        j = i + 1
        while j < len(parsed) and parsed[j][0] == season and parsed[j][1] == end + 1:
            end = parsed[j][1]
            j += 1
        if start == end:
            out.append(f"S{season:02d}E{start:02d}")
        else:
            out.append(f"S{season:02d}E{start:02d}-E{end:02d}")
        i = j
    return ", ".join(out)


def tvmaze_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "Just-TV-UI/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _norm_header(v: str) -> str:
    return "".join(ch for ch in str(v or "").strip().lower() if ch.isalnum())


def _pick_header(headers: List[str], names: List[str]) -> Optional[str]:
    hmap = {_norm_header(h): h for h in headers}
    for n in names:
        key = _norm_header(n)
        if key in hmap:
            return hmap[key]
    return None


def _is_tv_like(row: sqlite3.Row, has_is_tv_col: bool) -> bool:
    if has_is_tv_col:
        return int(row["is_tv_series"] or 0) == 1
    return bool(re.search(r"(season|series|complete|volume|vol\.?|book|disc)", str(row["title"] or ""), re.IGNORECASE))


def get_clz_csv_path() -> Optional[Path]:
    cfg = load_app_cfg()
    configured = normalize_str(cfg.get("clz_export_path"))
    if configured:
        p = Path(configured)
        if p.exists():
            return p
    if CLZ_CSV.exists():
        return CLZ_CSV
    return None


def import_refresh_clz() -> Tuple[int, bool]:
    clz_path = get_clz_csv_path()
    if not clz_path:
        return 0, False
    with clz_path.open("r", encoding="utf-8-sig", newline="") as f, get_conn() as conn:
        rdr = csv.DictReader(f)
        headers = rdr.fieldnames or []
        title_h = _pick_header(headers, ["Title"])
        idx_h = _pick_header(headers, ["Index", "CLZ Index", "Nr", "Number"])
        barcode_h = _pick_header(headers, ["Barcode", "UPC", "Upc"])
        box_h = _pick_header(headers, ["Box Set", "Boxset"])
        is_tv_h = _pick_header(headers, ["Is TV Series", "TV Series", "IsTvSeries"])
        discs_h = _pick_header(headers, ["Nr Discs", "Discs", "Disc Count"])
        format_h = _pick_header(headers, ["Format"])
        imdb_h = _pick_header(headers, ["IMDb Url", "IMDB Url", "IMDb URL"])
        notes_h = _pick_header(headers, ["Notes"])
        if not title_h:
            return 0, is_tv_h is not None
        count = 0
        for i, row in enumerate(rdr, start=1):
            title = normalize_str(row.get(title_h))
            if not title:
                continue
            clz_index = normalize_str(row.get(idx_h)) if idx_h else ""
            clz_index = clz_index or str(i)
            conn.execute(
                """
                INSERT INTO clz_item(clz_index,title,barcode,box_set,is_tv_series,nr_discs,format,imdb_url,notes,raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(clz_index) DO UPDATE SET
                  title=excluded.title,barcode=excluded.barcode,box_set=excluded.box_set,is_tv_series=excluded.is_tv_series,
                  nr_discs=excluded.nr_discs,format=excluded.format,imdb_url=excluded.imdb_url,notes=excluded.notes,raw_json=excluded.raw_json
                """,
                (
                    clz_index,
                    title,
                    normalize_str(row.get(barcode_h)) if barcode_h else "",
                    normalize_str(row.get(box_h)) if box_h else "",
                    1 if normalize_str(row.get(is_tv_h)).lower() in {"1", "true", "yes", "y", "t"} else 0,
                    parse_int(row.get(discs_h), 1) or 1,
                    normalize_str(row.get(format_h)) if format_h else "",
                    normalize_str(row.get(imdb_h)) if imdb_h else "",
                    normalize_str(row.get(notes_h)) if notes_h else "",
                    json.dumps(row, ensure_ascii=False),
                ),
            )
            count += 1
        return count, is_tv_h is not None


def load_plan_rows(search: str, tv_only: bool, group_sets: bool) -> Dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM clz_item ORDER BY CAST(clz_index AS INTEGER), clz_index").fetchall()
    total = len(rows)
    app_cfg = read_json(APP_CFG, default={})
    has_is_tv_col = bool(app_cfg.get("has_is_tv_column", False)) if isinstance(app_cfg, dict) else False
    tv_count = 0
    q = normalize_str(search).lower()
    filtered: List[sqlite3.Row] = []
    for r in rows:
        is_tv = _is_tv_like(r, has_is_tv_col)
        if is_tv:
            tv_count += 1
        if tv_only and not is_tv:
            continue
        if q and q not in str(r["title"] or "").lower():
            continue
        filtered.append(r)

    by_title = {str(r["title"] or "").strip().lower(): r for r in filtered}
    parent_labels: Dict[str, str] = {}
    parent_to_children: Dict[str, List[sqlite3.Row]] = {r["clz_index"]: [] for r in filtered}
    roots: List[sqlite3.Row] = []
    for r in filtered:
        parent_idx = normalize_str(r["parent_index"])
        if not parent_idx and normalize_str(r["box_set"]):
            maybe = by_title.get(normalize_str(r["box_set"]).lower())
            if maybe and maybe["clz_index"] != r["clz_index"]:
                parent_idx = maybe["clz_index"]
        if parent_idx and parent_idx in parent_to_children and parent_idx != r["clz_index"] and group_sets:
            parent_to_children[parent_idx].append(r)
            p = by_title.get(str(parent_idx).lower())
            parent_labels[r["clz_index"]] = str(parent_idx)
        else:
            roots.append(r)

    return {
        "rows": filtered,
        "roots": roots,
        "parent_to_children": parent_to_children,
        "parent_labels": parent_labels,
        "total_rows": total,
        "tv_filtered_rows": tv_count,
        "has_is_tv_column": has_is_tv_col,
    }


def get_package_row(clz_index: str) -> Optional[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM clz_item WHERE clz_index=?", (clz_index,)).fetchone()


def get_package_link(clz_index: str) -> Dict[str, Any]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM package_link WHERE clz_index=?", (clz_index,)).fetchone()
    return dict(row) if row else {}


def get_cached_episodes(tvmaze_show_id: int, season: Optional[int] = None) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        if season is None:
            rows = conn.execute(
                "SELECT * FROM episode_cache WHERE tvmaze_show_id=? ORDER BY season, episode_number",
                (tvmaze_show_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM episode_cache WHERE tvmaze_show_id=? AND season=? ORDER BY episode_number",
                (tvmaze_show_id, int(season)),
            ).fetchall()
    return [dict(r) for r in rows]


def refresh_show_episodes(tvmaze_show_id: int) -> int:
    data = tvmaze_get_json(f"https://api.tvmaze.com/shows/{int(tvmaze_show_id)}/episodes")
    if not isinstance(data, list):
        return 0
    count = 0
    with get_conn() as conn:
        now = now_iso()
        for e in data:
            season = parse_int(e.get("season"), 0)
            epnum = parse_int(e.get("number"), 0)
            if season <= 0 or epnum <= 0:
                continue
            code = f"S{season:02d}E{epnum:02d}"
            conn.execute(
                """
                INSERT INTO episode_cache(tvmaze_show_id,season,episode_number,episode_code,episode_title,tvmaze_episode_id,runtime_minutes,payload_json,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(tvmaze_show_id,season,episode_number) DO UPDATE SET
                  episode_code=excluded.episode_code,
                  episode_title=excluded.episode_title,
                  tvmaze_episode_id=excluded.tvmaze_episode_id,
                  runtime_minutes=excluded.runtime_minutes,
                  payload_json=excluded.payload_json,
                  updated_at=excluded.updated_at
                """,
                (
                    int(tvmaze_show_id),
                    season,
                    epnum,
                    code,
                    normalize_str(e.get("name")),
                    parse_int(e.get("id"), 0) or None,
                    parse_int(e.get("runtime"), 0) or None,
                    json.dumps(e, ensure_ascii=False),
                    now,
                ),
            )
            count += 1
    return count


def get_package_discs(clz_index: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM disc WHERE clz_index=? ORDER BY disc_number", (clz_index,)).fetchall()
    return [dict(r) for r in rows]


def get_disc_episode_maps(disc_uid: str) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM disc_episode_map WHERE disc_uid=? ORDER BY season, episode_number, episode_code",
            (disc_uid,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_disc_by_uid(disc_uid: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM disc WHERE disc_uid=?", (disc_uid,)).fetchone()
    return dict(row) if row else None


def get_mapped_disc_candidates(min_clip_seconds: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.disc_uid, d.clz_index, d.disc_number, d.label,
                   c.title AS package_title,
                   COALESCE(pl.show_name, c.title) AS series_name,
                   COALESCE(pl.selected_season, 0) AS selected_season,
                   COUNT(m.episode_code) AS mapped_count
            FROM disc d
            JOIN clz_item c ON c.clz_index = d.clz_index
            LEFT JOIN package_link pl ON pl.clz_index = d.clz_index
            LEFT JOIN disc_episode_map m ON m.disc_uid = d.disc_uid
            GROUP BY d.disc_uid, d.clz_index, d.disc_number, d.label, c.title, pl.show_name, pl.selected_season
            ORDER BY LOWER(COALESCE(pl.show_name, c.title)), COALESCE(pl.selected_season, 0), d.disc_number
            """
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        season = parse_int(r["selected_season"], 0)
        queue_key = make_queue_key(str(r["series_name"] or ""), season, int(r["disc_number"] or 0))
        eps = get_disc_episode_maps(r["disc_uid"])
        short_count = sum(1 for e in eps if parse_int(e.get("runtime_minutes"), 0) > 0 and parse_int(e.get("runtime_minutes"), 0) * 60 < min_clip_seconds)
        out.append(
            {
                "queue_key": queue_key,
                "series": str(r["series_name"] or ""),
                "season": season,
                "disc": int(r["disc_number"] or 0),
                "episode_count": int(r["mapped_count"] or 0),
                "short_count": short_count,
                "include_flag": 1,
                "priority": i,
                "episodes": [
                    {
                        "sxxeyy": e.get("episode_code"),
                        "min_minutes": parse_int(e.get("runtime_minutes"), 0) or None,
                        "max_minutes": parse_int(e.get("runtime_minutes"), 0) or None,
                    }
                    for e in eps
                ],
                "disc_uid": r["disc_uid"],
                "package_title": r["package_title"],
            }
        )
    return out


def map_rows(min_clip_seconds: int) -> List[Dict[str, Any]]:
    base = get_mapped_disc_candidates(min_clip_seconds)
    with get_conn() as conn:
        stored = {r["queue_key"]: dict(r) for r in conn.execute("SELECT * FROM mapped_item").fetchall()}
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(base, start=1):
        st = stored.get(d["queue_key"], {})
        out.append(
            {
                **d,
                "include_flag": int(st.get("include_flag", d.get("include_flag", 1))),
                "priority": int(st.get("priority", d.get("priority", i))),
            }
        )
    out.sort(key=lambda r: (r["priority"], str(r["series"]).lower(), r["season"], r["disc"]))
    return out


def save_mapping(rows: List[Dict[str, Any]]) -> None:
    with get_conn() as conn:
        now = now_iso()
        for r in rows:
            conn.execute(
                """
                INSERT INTO mapped_item(queue_key, include_flag, priority, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(queue_key) DO UPDATE SET
                  include_flag=excluded.include_flag,
                  priority=excluded.priority,
                  updated_at=excluded.updated_at
                """,
                (r["queue_key"], int(r["include_flag"]), int(r["priority"]), now),
            )


def build_queue_from_mapping(rows: List[Dict[str, Any]]) -> int:
    keys = [r["queue_key"] for r in sorted(rows, key=lambda x: x["priority"]) if int(r["include_flag"]) == 1]
    payload = {
        "built_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "index": str(TV_INDEX),
        "queue_keys": keys,
    }
    write_json(TV_QUEUE, payload)
    return len(keys)


def load_queue_draft() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT queue_key, position, included, priority, added_at FROM queue_draft ORDER BY position, priority, queue_key"
        ).fetchall()
    return [dict(r) for r in rows]


def add_to_queue_draft(keys: List[str]) -> int:
    clean = [normalize_str(k) for k in keys if normalize_str(k)]
    if not clean:
        return 0
    added = 0
    with get_conn() as conn:
        row = conn.execute("SELECT COALESCE(MAX(position), 0) AS mx FROM queue_draft").fetchone()
        pos = int(row["mx"] or 0)
        now = now_iso()
        for k in clean:
            exists = conn.execute("SELECT 1 FROM queue_draft WHERE queue_key=?", (k,)).fetchone()
            if exists:
                continue
            pos += 1
            conn.execute(
                "INSERT INTO queue_draft(queue_key, position, included, priority, added_at) VALUES (?, ?, 1, ?, ?)",
                (k, pos, pos, now),
            )
            added += 1
    return added


def clear_queue_draft() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM queue_draft")


def remove_from_queue_draft(queue_key: str) -> None:
    key = normalize_str(queue_key)
    if not key:
        return
    with get_conn() as conn:
        conn.execute("DELETE FROM queue_draft WHERE queue_key=?", (key,))
        _normalize_queue_positions_conn(conn)


def _normalize_queue_positions_conn(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT queue_key FROM queue_draft ORDER BY position, priority, queue_key").fetchall()
    for i, r in enumerate(rows, start=1):
        conn.execute("UPDATE queue_draft SET position=?, priority=? WHERE queue_key=?", (i, i, r["queue_key"]))


def move_queue_draft(queue_key: str, delta: int) -> None:
    key = normalize_str(queue_key)
    if not key or delta == 0:
        return
    with get_conn() as conn:
        rows = conn.execute("SELECT queue_key FROM queue_draft ORDER BY position, priority, queue_key").fetchall()
        keys = [r["queue_key"] for r in rows]
        if key not in keys:
            return
        i = keys.index(key)
        j = i + delta
        if j < 0 or j >= len(keys):
            return
        keys[i], keys[j] = keys[j], keys[i]
        for pos, qk in enumerate(keys, start=1):
            conn.execute("UPDATE queue_draft SET position=?, priority=? WHERE queue_key=?", (pos, pos, qk))


def save_queue_draft_from_form(form: Any) -> int:
    keys = [normalize_str(k) for k in form.getlist("queue_key_list") if normalize_str(k)]
    include = set(form.getlist("include_q"))
    parsed: List[Tuple[int, str, int]] = []
    for i, key in enumerate(keys, start=1):
        pos = parse_int(form.get(f"position__{key}"), i)
        parsed.append((pos, key, 1 if key in include else 0))
    parsed.sort(key=lambda x: (x[0], x[1]))
    with get_conn() as conn:
        now = now_iso()
        for i, (_, key, inc) in enumerate(parsed, start=1):
            conn.execute(
                """
                INSERT INTO queue_draft(queue_key, position, included, priority, added_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(queue_key) DO UPDATE SET
                  position=excluded.position,
                  included=excluded.included,
                  priority=excluded.priority
                """,
                (key, i, inc, i, now),
            )
        if parsed:
            placeholders = ",".join(["?"] * len(parsed))
            keep = [k for _, k, _ in parsed]
            conn.execute(f"DELETE FROM queue_draft WHERE queue_key NOT IN ({placeholders})", keep)
        else:
            conn.execute("DELETE FROM queue_draft")
    return len(parsed)


def build_queue_from_draft() -> int:
    rows = load_queue_draft()
    keys = [r["queue_key"] for r in rows if int(r.get("included") or 0) == 1]
    payload = {
        "built_at": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "index": str(TV_INDEX),
        "queue_keys": keys,
    }
    write_json(TV_QUEUE, payload)
    return len(keys)


def set_run_state(**kwargs: Any) -> None:
    with RUN_LOCK:
        RUN_STATE_MEM.update(kwargs)
        RUN_STATE_MEM["last_update"] = now_iso()
        write_json(RUN_STATE, RUN_STATE_MEM)


def tail_log(n: int = 80) -> List[str]:
    if not RUN_LOG.exists():
        return []
    return RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def monitor(proc: subprocess.Popen) -> None:
    ensure_dir(RUN_LOG.parent)
    with RUN_LOG.open("a", encoding="utf-8", errors="replace") as f:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    break
                continue
            txt = line.rstrip("\r\n")
            f.write(txt + "\n")
            f.flush()
            if "NEXT DISC:" in txt:
                set_run_state(current_step="waiting_for_disc", current_item=txt.split("NEXT DISC:", 1)[1].strip())
            elif "Dirty mode: ripping" in txt:
                set_run_state(current_step="ripping")
    rc = proc.poll()
    if rc and int(rc) != 0:
        tail = tail_log(10)
        err = next((ln for ln in reversed(tail) if normalize_str(ln)), f"Process exited with code {rc}")
        set_run_state(running=False, current_step="error", returncode=rc, return_code=rc, error=err)
    else:
        set_run_state(running=False, current_step="finished", returncode=rc, return_code=rc, error=None)
    with RUN_LOCK:
        global RUN_PROC
        RUN_PROC = None


@app.on_event("startup")
def startup() -> None:
    init_db()
    cfg = load_app_cfg()
    if not APP_CFG.exists():
        save_cfg(cfg)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/plan", status_code=302)


@app.get("/plan")
def plan_page(request: Request, q: str = "", tv_only: int = 1, group_sets: int = 1, msg: str = ""):
    info = load_plan_rows(q, bool(tv_only), bool(group_sets))
    clz_path = get_clz_csv_path()
    cfg = load_app_cfg()
    configured_clz = normalize_str(cfg.get("clz_export_path"))
    return TEMPLATES.TemplateResponse(
        "plan.html",
        {
            "request": request,
            "msg": msg,
            "q": q,
            "tv_only": int(bool(tv_only)),
            "group_sets": int(bool(group_sets)),
            "active_clz_path": str(clz_path) if clz_path else (configured_clz or str(CLZ_CSV)),
            "active_clz_exists": bool(clz_path and clz_path.exists()),
            "configured_clz_path": configured_clz,
            **nav_ctx("plan"),
            **workflow_ctx(),
            **info,
        },
    )


@app.post("/plan/import-clz")
def plan_import_clz():
    count, has_col = import_refresh_clz()
    app_cfg = load_app_cfg()
    app_cfg["has_is_tv_column"] = bool(has_col)
    write_json(APP_CFG, app_cfg)
    if count == 0 and get_clz_csv_path() is None:
        return RedirectResponse(url="/plan?msg=Missing%20CLZ%20export", status_code=303)
    return RedirectResponse(url=f"/plan?msg=Imported%20{count}%20CLZ%20rows", status_code=303)


@app.post("/plan/pick-clz")
def plan_pick_clz():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select CLZ Export CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        root.destroy()
        if not path:
            return RedirectResponse(url="/plan?msg=No%20file%20selected", status_code=303)
        cfg = load_app_cfg()
        cfg["clz_export_path"] = path
        save_cfg(cfg)
        return RedirectResponse(url="/plan?msg=Selected%20CLZ%20CSV", status_code=303)
    except Exception:
        return RedirectResponse(url="/plan?msg=Picker%20not%20available%20(use%20/config%20to%20set%20path)", status_code=303)


@app.post("/plan/build-index")
def plan_build_index():
    if not TV_MANIFEST.exists():
        return RedirectResponse(url="/plan?msg=Missing%20TV%20Manifest", status_code=303)
    try:
        tv = importlib.import_module("discmapper_tv_v02_1")
        idx = tv.build_tv_index(TV_MANIFEST)
        tv.write_json(TV_INDEX, idx)
        return RedirectResponse(url="/plan?msg=Built%20TV%20Index.%20Next:%20click%20a%20Title%20to%20map%20discs/episodes", status_code=303)
    except Exception as e:
        return RedirectResponse(url=f"/plan?msg=Build%20index%20failed:%20{str(e)}", status_code=303)


@app.post("/plan/export-manifest")
def plan_export_manifest():
    if TV_MANIFEST.exists():
        return RedirectResponse(url="/plan?msg=TV%20Manifest%20already%20present", status_code=303)
    return RedirectResponse(url="/plan?msg=No%20TV%20Manifest%20generator%20in%20this%20repo.%20Provide%20TV%20Manifest", status_code=303)


@app.post("/plan/export-queue")
def plan_export_queue():
    rows = [r for r in map_rows(int(load_app_cfg()["min_clip_seconds"])) if int(r.get("episode_count") or 0) > 0]
    if not rows:
        return RedirectResponse(url="/plan?msg=No%20mapped%20rows%20to%20queue", status_code=303)
    for i, r in enumerate(rows, start=1):
        r["priority"] = i
        r["include_flag"] = 1
    save_mapping(rows)
    count = build_queue_from_mapping(rows)
    return RedirectResponse(url=f"/plan?msg=Built%20Rip%20Queue%20({count})", status_code=303)


@app.get("/plan/validate")
def plan_validate():
    queue = read_json(TV_QUEUE, default={})
    queue_count = len(queue.get("queue_keys") or []) if isinstance(queue, dict) else 0
    return JSONResponse(
        {
            "manifest_exists": TV_MANIFEST.exists(),
            "index_exists": TV_INDEX.exists(),
            "queue_exists": TV_QUEUE.exists(),
            "queue_count": queue_count,
            "clz_exists": get_clz_csv_path() is not None,
            "clz_path": str(get_clz_csv_path()) if get_clz_csv_path() else None,
        }
    )


@app.post("/plan/parent/{clz_index}")
def set_parent(clz_index: str, parent_index: str = Form(default="")):
    with get_conn() as conn:
        conn.execute("UPDATE clz_item SET parent_index=? WHERE clz_index=?", (normalize_str(parent_index) or None, clz_index))
    return RedirectResponse(url=f"/plan?msg=Parent%20saved%20for%20{clz_index}", status_code=303)


@app.post("/plan/parent-clear/{clz_index}")
def clear_parent(clz_index: str):
    with get_conn() as conn:
        conn.execute("UPDATE clz_item SET parent_index=NULL WHERE clz_index=?", (clz_index,))
    return RedirectResponse(url=f"/plan?msg=Parent%20cleared%20for%20{clz_index}", status_code=303)


@app.get("/package/{clz_index}")
def package_page(request: Request, clz_index: str, q: str = "", msg: str = ""):
    pkg = get_package_row(clz_index)
    if not pkg:
        return RedirectResponse(url="/plan?msg=Package%20not%20found", status_code=303)
    link = get_package_link(clz_index)
    show_id = parse_int(link.get("tvmaze_show_id"), 0)
    selected_season = parse_int(link.get("selected_season"), 0)
    seasons: List[int] = []
    episodes: List[Dict[str, Any]] = []
    if show_id > 0:
        all_eps = get_cached_episodes(show_id)
        seasons = sorted({parse_int(e.get("season"), 0) for e in all_eps if parse_int(e.get("season"), 0) > 0})
        if selected_season > 0:
            episodes = get_cached_episodes(show_id, selected_season)
        elif all_eps:
            selected_season = parse_int(all_eps[0].get("season"), 0)
            episodes = get_cached_episodes(show_id, selected_season)

    search_results: List[Dict[str, Any]] = []
    if normalize_str(q):
        try:
            resp = tvmaze_get_json(f"https://api.tvmaze.com/search/shows?q={urllib.parse.quote(normalize_str(q))}")
            if isinstance(resp, list):
                for item in resp[:20]:
                    show = item.get("show") if isinstance(item, dict) else {}
                    if not isinstance(show, dict):
                        continue
                    ext = show.get("externals") if isinstance(show.get("externals"), dict) else {}
                    search_results.append(
                        {
                            "id": parse_int(show.get("id"), 0),
                            "name": normalize_str(show.get("name")),
                            "premiered": normalize_str(show.get("premiered")),
                            "imdb_id": normalize_str(ext.get("imdb")),
                        }
                    )
        except Exception:
            search_results = []

    discs = get_package_discs(clz_index)
    episodes_assigned = False
    for d in discs:
        d["mapped"] = get_disc_episode_maps(d["disc_uid"])
        d["mapped_count"] = len(d["mapped"])
        d["mapped_summary"] = summarize_episode_codes([normalize_str(m.get("episode_code")) for m in d["mapped"] if normalize_str(m.get("episode_code"))])
        if d["mapped_count"] > 0:
            episodes_assigned = True
    show_linked = show_id > 0
    episodes_cached = bool(show_id > 0 and len(get_cached_episodes(show_id)) > 0)
    season_selected = selected_season > 0
    discs_created = len(discs) > 0
    checklist = [
        ("Show linked", show_linked),
        ("Episodes cached", episodes_cached),
        ("Season selected", season_selected),
        ("Discs created", discs_created),
        ("Episodes assigned to discs", episodes_assigned),
    ]
    package_next_step = "Mapping checklist complete."
    if not show_linked:
        package_next_step = "Next: Link a TVMaze show."
    elif not episodes_cached:
        package_next_step = "Next: Click Refresh Episodes."
    elif not season_selected:
        package_next_step = "Next: Select and save a season."
    elif not discs_created:
        package_next_step = "Next: Create discs for this package."
    elif not episodes_assigned:
        package_next_step = "Next: Open a disc and assign episodes."

    return TEMPLATES.TemplateResponse(
        "package.html",
        {
            "request": request,
            "msg": msg,
            "pkg": pkg,
            "link": link,
            "seasons": seasons,
            "selected_season": selected_season,
            "episodes": episodes,
            "search_q": q,
            "search_results": search_results,
            "discs": discs,
            "episodes_cached": episodes_cached,
            "mapping_checklist": checklist,
            "package_next_step": package_next_step,
            **nav_ctx("package", clz_index),
            **workflow_ctx(),
        },
    )


@app.post("/package/{clz_index}/show-link")
async def package_show_link(request: Request, clz_index: str):
    form = await request.form()
    show_id = parse_int(form.get("tvmaze_show_id"), 0)
    show_name = normalize_str(form.get("show_name"))
    imdb_id = normalize_str(form.get("imdb_id"))
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO package_link(clz_index,tvmaze_show_id,show_name,imdb_id,selected_season,updated_at)
            VALUES (?,?,?,?,NULL,?)
            ON CONFLICT(clz_index) DO UPDATE SET
              tvmaze_show_id=excluded.tvmaze_show_id,
              show_name=excluded.show_name,
              imdb_id=excluded.imdb_id,
              updated_at=excluded.updated_at
            """,
            (clz_index, show_id or None, show_name, imdb_id, now_iso()),
        )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Linked%20show", status_code=303)


@app.post("/package/{clz_index}/parent")
async def package_set_parent(request: Request, clz_index: str):
    form = await request.form()
    parent_index = normalize_str(form.get("parent_index"))
    with get_conn() as conn:
        conn.execute("UPDATE clz_item SET parent_index=? WHERE clz_index=?", (parent_index or None, clz_index))
    return RedirectResponse(url=f"/package/{clz_index}?msg=Parent%20saved", status_code=303)


@app.post("/package/{clz_index}/parent-clear")
def package_clear_parent(clz_index: str):
    with get_conn() as conn:
        conn.execute("UPDATE clz_item SET parent_index=NULL WHERE clz_index=?", (clz_index,))
    return RedirectResponse(url=f"/package/{clz_index}?msg=Parent%20cleared", status_code=303)


@app.post("/package/{clz_index}/season")
async def package_set_season(request: Request, clz_index: str):
    form = await request.form()
    season = parse_int(form.get("selected_season"), 0)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO package_link(clz_index,selected_season,updated_at)
            VALUES (?,?,?)
            ON CONFLICT(clz_index) DO UPDATE SET
              selected_season=excluded.selected_season,
              updated_at=excluded.updated_at
            """,
            (clz_index, season or None, now_iso()),
        )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Season%20saved", status_code=303)


@app.post("/package/{clz_index}/episodes-refresh")
def package_refresh_episodes(clz_index: str):
    link = get_package_link(clz_index)
    show_id = parse_int(link.get("tvmaze_show_id"), 0)
    if show_id <= 0:
        return RedirectResponse(url=f"/package/{clz_index}?msg=Link%20a%20show%20first", status_code=303)
    try:
        count = refresh_show_episodes(show_id)
        return RedirectResponse(url=f"/package/{clz_index}?msg=Refreshed%20{count}%20episodes", status_code=303)
    except Exception as e:
        return RedirectResponse(url=f"/package/{clz_index}?msg=Episode%20refresh%20failed:%20{str(e)}", status_code=303)


@app.post("/package/{clz_index}/discs-reset")
async def package_discs_reset(request: Request, clz_index: str):
    form = await request.form()
    n = max(1, parse_int(form.get("disc_count"), 1))
    with get_conn() as conn:
        now = now_iso()
        for d in range(1, n + 1):
            uid = make_disc_uid(clz_index, d)
            conn.execute(
                """
                INSERT INTO disc(disc_uid,clz_index,disc_number,label,updated_at)
                VALUES (?,?,?,?,?)
                ON CONFLICT(clz_index,disc_number) DO UPDATE SET
                  updated_at=excluded.updated_at
                """,
                (uid, clz_index, d, None, now),
            )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Discs%20reset%20to%20{n}", status_code=303)


@app.post("/package/{clz_index}/disc-save")
async def package_disc_save(request: Request, clz_index: str):
    form = await request.form()
    disc_number = max(1, parse_int(form.get("disc_number"), 1))
    label = normalize_str(form.get("label"))
    uid = make_disc_uid(clz_index, disc_number)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO disc(disc_uid,clz_index,disc_number,label,updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(clz_index,disc_number) DO UPDATE SET
              disc_uid=excluded.disc_uid,
              label=excluded.label,
              updated_at=excluded.updated_at
            """,
            (uid, clz_index, disc_number, label or None, now_iso()),
        )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Disc%20saved", status_code=303)


@app.post("/package/{clz_index}/disc-assign")
async def package_disc_assign(request: Request, clz_index: str):
    form = await request.form()
    disc_number = max(1, parse_int(form.get("disc_number"), 1))
    uid = make_disc_uid(clz_index, disc_number)
    code_text = normalize_str(form.get("episode_codes"))
    selected_codes = [normalize_str(v).upper() for v in form.getlist("episode_code") if normalize_str(v)]
    parsed_codes = parse_episode_codes(code_text)
    codes = selected_codes + [c for c in parsed_codes if c not in selected_codes]
    link = get_package_link(clz_index)
    show_id = parse_int(link.get("tvmaze_show_id"), 0)
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO disc(disc_uid,clz_index,disc_number,updated_at) VALUES (?,?,?,?)", (uid, clz_index, disc_number, now_iso()))
        conn.execute("DELETE FROM disc_episode_map WHERE disc_uid=?", (uid,))
        now = now_iso()
        for code in codes:
            row = None
            if show_id > 0:
                row = conn.execute(
                    "SELECT * FROM episode_cache WHERE tvmaze_show_id=? AND UPPER(episode_code)=?",
                    (show_id, code.upper()),
                ).fetchone()
            if row:
                conn.execute(
                    """
                    INSERT INTO disc_episode_map(disc_uid,episode_code,tvmaze_episode_id,season,episode_number,episode_title,runtime_minutes,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        uid,
                        normalize_str(row["episode_code"]).upper(),
                        parse_int(row["tvmaze_episode_id"], 0) or None,
                        parse_int(row["season"], 0) or None,
                        parse_int(row["episode_number"], 0) or None,
                        normalize_str(row["episode_title"]),
                        parse_int(row["runtime_minutes"], 0) or None,
                        now,
                    ),
                )
            else:
                parsed = parse_episode_code_token(code)
                season = parsed[0] if parsed else None
                epnum = parsed[1] if parsed else None
                conn.execute(
                    """
                    INSERT INTO disc_episode_map(disc_uid,episode_code,tvmaze_episode_id,season,episode_number,episode_title,runtime_minutes,updated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (uid, code.upper(), None, season, epnum, "", None, now),
                )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Disc%20episode%20mapping%20saved", status_code=303)


@app.get("/disc/{disc_uid}")
def disc_page(request: Request, disc_uid: str, msg: str = ""):
    disc = get_disc_by_uid(disc_uid)
    if not disc:
        return RedirectResponse(url="/plan?msg=Disc%20not%20found", status_code=303)
    clz_index = normalize_str(disc.get("clz_index"))
    pkg = get_package_row(clz_index)
    link = get_package_link(clz_index)
    show_id = parse_int(link.get("tvmaze_show_id"), 0)
    selected_season = parse_int(link.get("selected_season"), 0)
    episodes: List[Dict[str, Any]] = []
    if show_id > 0 and selected_season > 0:
        episodes = get_cached_episodes(show_id, selected_season)
    mapped = get_disc_episode_maps(disc_uid)
    mapped_codes = {normalize_str(m.get("episode_code")).upper() for m in mapped if normalize_str(m.get("episode_code"))}
    return TEMPLATES.TemplateResponse(
        "disc.html",
        {
            "request": request,
            "msg": msg,
            "disc": disc,
            "pkg": pkg,
            "link": link,
            "selected_season": selected_season,
            "episodes": episodes,
            "mapped_codes": mapped_codes,
            **nav_ctx("package", clz_index),
        },
    )


@app.post("/disc/{disc_uid}/save")
async def disc_save(disc_uid: str, request: Request):
    disc = get_disc_by_uid(disc_uid)
    if not disc:
        return RedirectResponse(url="/plan?msg=Disc%20not%20found", status_code=303)
    clz_index = normalize_str(disc.get("clz_index"))
    link = get_package_link(clz_index)
    show_id = parse_int(link.get("tvmaze_show_id"), 0)
    season = parse_int(link.get("selected_season"), 0)
    form = await request.form()
    selected_codes = [normalize_str(v).upper() for v in form.getlist("episode_code") if normalize_str(v)]
    with get_conn() as conn:
        conn.execute("DELETE FROM disc_episode_map WHERE disc_uid=?", (disc_uid,))
        now = now_iso()
        # preserve stable insertion order by episode number within selected season
        order_rows = []
        if show_id > 0 and season > 0:
            for code in selected_codes:
                row = conn.execute(
                    "SELECT * FROM episode_cache WHERE tvmaze_show_id=? AND season=? AND UPPER(episode_code)=?",
                    (show_id, season, code),
                ).fetchone()
                if row:
                    order_rows.append(dict(row))
        order_rows.sort(key=lambda r: (parse_int(r.get("episode_number"), 0), normalize_str(r.get("episode_code"))))
        used_codes = set()
        for row in order_rows:
            code = normalize_str(row.get("episode_code")).upper()
            used_codes.add(code)
            conn.execute(
                """
                INSERT INTO disc_episode_map(disc_uid,episode_code,tvmaze_episode_id,season,episode_number,episode_title,runtime_minutes,updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    disc_uid,
                    code,
                    parse_int(row.get("tvmaze_episode_id"), 0) or None,
                    parse_int(row.get("season"), 0) or None,
                    parse_int(row.get("episode_number"), 0) or None,
                    normalize_str(row.get("episode_title")),
                    parse_int(row.get("runtime_minutes"), 0) or None,
                    now,
                ),
            )
        for code in selected_codes:
            if code in used_codes:
                continue
            parsed = parse_episode_code_token(code)
            conn.execute(
                """
                INSERT INTO disc_episode_map(disc_uid,episode_code,tvmaze_episode_id,season,episode_number,episode_title,runtime_minutes,updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    disc_uid,
                    code,
                    None,
                    parsed[0] if parsed else None,
                    parsed[1] if parsed else None,
                    "",
                    None,
                    now,
                ),
            )
    return RedirectResponse(url=f"/package/{clz_index}?msg=Saved%20Disc%20{disc_uid}", status_code=303)


@app.get("/map")
def map_page(request: Request, show_short: int = 0, msg: str = ""):
    cfg = load_app_cfg()
    rows = map_rows(int(cfg["min_clip_seconds"]))
    if not show_short:
        for r in rows:
            r["episodes"] = [e for e in r["episodes"] if not (isinstance(e.get("min_minutes"), (int, float)) and int(e["min_minutes"] * 60) < int(cfg["min_clip_seconds"]))]
    return TEMPLATES.TemplateResponse("map.html", {"request": request, "rows": rows, "cfg": cfg, "show_short": int(show_short), "msg": msg, **nav_ctx("summary"), **workflow_ctx()})


@app.post("/map/save")
async def map_save(request: Request):
    form = await request.form()
    cfg = load_app_cfg()
    rows = map_rows(int(cfg["min_clip_seconds"]))
    include = set(form.getlist("include"))
    parsed: List[Dict[str, Any]] = []
    for r in rows:
        pos = int(form.get(f"priority__{r['queue_key']}", r["priority"]))
        parsed.append({**r, "include_flag": 1 if r["queue_key"] in include else 0, "priority": pos})
    parsed.sort(key=lambda x: x["priority"])
    for i, r in enumerate(parsed, start=1):
        r["priority"] = i
    save_mapping(parsed)
    return RedirectResponse(url="/map?msg=Mapping%20saved", status_code=303)


@app.get("/queue")
def queue_page(request: Request, msg: str = "", show_unmapped: int = 0):
    cfg = load_app_cfg()
    rows = map_rows(int(cfg["min_clip_seconds"]))
    candidates: List[Dict[str, Any]] = []
    mapped_count = 0
    for r in rows:
        r["is_mapped"] = int(r.get("episode_count") or 0) > 0
        if r["is_mapped"]:
            mapped_count += 1
        if (not show_unmapped) and (not r["is_mapped"]):
            continue
        candidates.append(r)

    draft_raw = load_queue_draft()
    by_key = {r["queue_key"]: r for r in rows}
    queue_rows: List[Dict[str, Any]] = []
    for i, d in enumerate(draft_raw, start=1):
        meta = by_key.get(d["queue_key"], {})
        queue_rows.append(
            {
                "position": int(d.get("position") or i),
                "included": int(d.get("included") or 0),
                "queue_key": d["queue_key"],
                "series": meta.get("series"),
                "season": meta.get("season"),
                "disc": meta.get("disc"),
                "episode_count": meta.get("episode_count", 0),
                "short_count": meta.get("short_count", 0),
                "missing_meta": meta == {},
            }
        )
    queue_rows.sort(key=lambda x: (x["position"], x["queue_key"]))
    return TEMPLATES.TemplateResponse(
        "queue.html",
        {
            "request": request,
            "msg": msg,
            "queue_path": str(TV_QUEUE),
            "index_path": str(TV_INDEX),
            "candidates": candidates,
            "queue_rows": queue_rows,
            "show_unmapped": int(bool(show_unmapped)),
            "debug": {
                "candidate_source": "SQLite: disc + disc_episode_map",
                "total_candidates": len(rows),
                "mapped_candidates": mapped_count,
                "visible_candidates": len(candidates),
                "draft_count": len(queue_rows),
            },
            **nav_ctx("queue"),
            **workflow_ctx(),
        },
    )


@app.post("/queue/add")
async def queue_add(request: Request):
    form = await request.form()
    keys = form.getlist("candidate_key")
    added = add_to_queue_draft(keys)
    return RedirectResponse(url=f"/queue?msg=Added%20{added}%20item(s)%20to%20queue", status_code=303)


@app.post("/queue/save-draft")
async def queue_save_draft(request: Request):
    form = await request.form()
    count = save_queue_draft_from_form(form)
    return RedirectResponse(url=f"/queue?msg=Disc%20order%20saved%20({count}%20items)", status_code=303)


@app.post("/queue/remove")
async def queue_remove(request: Request):
    form = await request.form()
    key = normalize_str(form.get("action_key"))
    remove_from_queue_draft(key)
    return RedirectResponse(url="/queue?msg=Queue%20item%20removed", status_code=303)


@app.post("/queue/move-up")
async def queue_move_up(request: Request):
    form = await request.form()
    key = normalize_str(form.get("action_key"))
    move_queue_draft(key, -1)
    return RedirectResponse(url="/queue?msg=Queue%20item%20moved%20up", status_code=303)


@app.post("/queue/move-down")
async def queue_move_down(request: Request):
    form = await request.form()
    key = normalize_str(form.get("action_key"))
    move_queue_draft(key, 1)
    return RedirectResponse(url="/queue?msg=Queue%20item%20moved%20down", status_code=303)


@app.post("/queue/clear")
def queue_clear():
    clear_queue_draft()
    return RedirectResponse(url="/queue?msg=Queue%20draft%20cleared", status_code=303)


@app.post("/queue/build")
async def queue_build(request: Request):
    form = await request.form()
    save_queue_draft_from_form(form)
    count = build_queue_from_draft()
    return RedirectResponse(url=f"/queue?msg=Built%20Rip%20Queue:%20{count}%20discs%20-%3E%20{str(TV_QUEUE)}", status_code=303)


@app.get("/config")
def config_page(request: Request, msg: str = ""):
    cfg = load_app_cfg()
    v = validate_cfg(cfg)
    return TEMPLATES.TemplateResponse("config.html", {"request": request, "cfg": cfg, "errors": v["errors"], "warnings": v["warnings"], "msg": msg, **nav_ctx("config"), **workflow_ctx()})


@app.get("/run")
def run_page(request: Request, msg: str = ""):
    cfg = load_app_cfg()
    v = validate_cfg(cfg)
    queue = read_json(TV_QUEUE, default={})
    queue_count = len(queue.get("queue_keys") or []) if isinstance(queue, dict) else 0
    ready = (len(v["errors"]) == 0) and TV_QUEUE.exists() and queue_count > 0
    run_error = None
    if queue_count == 0:
        run_error = "Queue is empty. Go to /queue and build it."
    state = dict(RUN_STATE_MEM)
    state["running"] = bool(RUN_PROC is not None and RUN_PROC.poll() is None)
    return TEMPLATES.TemplateResponse(
        "run.html",
        {
            "request": request,
            "cfg": cfg,
            "errors": v["errors"],
            "warnings": v["warnings"],
            "queue_count": queue_count,
            "queue_exists": TV_QUEUE.exists(),
            "queue_path": str(TV_QUEUE),
            "engine_path": str(ROOT / "discmapper_tv_v02_1.py"),
            "ready": ready,
            "run_error": run_error,
            "msg": msg,
            "state": state,
            "log_tail": tail_log(),
            **nav_ctx("run"),
            **workflow_ctx(),
        },
    )


@app.get("/api/v1/config")
def api_get_config():
    cfg = load_app_cfg()
    v = validate_cfg(cfg)
    return JSONResponse({"config": cfg, "errors": v["errors"], "warnings": v["warnings"]})


@app.post("/api/v1/config/validate")
async def api_validate_config(request: Request):
    payload = await request.json()
    v = validate_cfg(payload)
    return JSONResponse({"ok": len(v["errors"]) == 0, "errors": v["errors"], "warnings": v["warnings"]})


@app.post("/api/v1/config/save")
async def api_save_config(request: Request):
    payload = await request.json()
    v = validate_cfg(payload)
    if v["errors"]:
        return JSONResponse({"ok": False, "errors": v["errors"], "warnings": v["warnings"]})
    cfg = save_cfg(payload)
    return JSONResponse({"ok": True, "config": cfg, "warnings": v["warnings"]})


@app.post("/api/v1/config/browse")
def api_browse():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        p = filedialog.askdirectory(); root.destroy()
        if not p:
            return JSONResponse({"ok": False, "error": "No folder selected"})
        return JSONResponse({"ok": True, "path": p})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Browse unavailable: {e}"})


@app.post("/api/v1/run/start")
def api_run_start():
    cfg = load_app_cfg()
    queue = read_json(TV_QUEUE, default={})
    queue_count = len(queue.get("queue_keys") or []) if isinstance(queue, dict) else 0
    prereq_errors: List[str] = []
    if not TV_QUEUE.exists():
        prereq_errors.append(f"Missing queue file: {TV_QUEUE}")
    if queue_count <= 0:
        prereq_errors.append("Queue is empty. Build it on /queue.")
    for key in ("rips_staging_root", "rips_complete_root"):
        p = Path(str(cfg.get(key) or ""))
        if not str(cfg.get(key) or "").strip() or not p.exists():
            prereq_errors.append(f"Missing required path: {key}={p}")
    engine_path = ROOT / "discmapper_tv_v02_1.py"
    if not engine_path.exists():
        prereq_errors.append(f"Missing engine entrypoint: {engine_path}")
    if not TV_INDEX.exists():
        prereq_errors.append(f"Missing index file: {TV_INDEX}")
    if prereq_errors:
        ensure_dir(LOG_DIR)
        ensure_dir(RUN_LOG.parent)
        msg = " | ".join(prereq_errors)
        with RUN_LOG.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"[{now_iso()}] START BLOCKED: {msg}\n")
        set_run_state(running=False, current_step="error", error=msg, returncode=None, return_code=None)
        return JSONResponse({"ok": False, "message": msg, "error": msg})

    with RUN_LOCK:
        global RUN_PROC
        if RUN_PROC is not None and RUN_PROC.poll() is None:
            return JSONResponse({"ok": False, "message": "Run already active"})

        ensure_dir(LOG_DIR)
        RUN_LOG.write_text("", encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "discmapper_tv_v02_1.py"), "rip-queue", "--index", str(TV_INDEX), "--queue", str(TV_QUEUE), "--config", str(TV_CFG)]
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        except Exception as e:
            msg = f"Failed to start engine: {e}"
            with RUN_LOG.open("a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{now_iso()}] START FAILED: {msg}\n")
            set_run_state(running=False, current_step="error", error=msg, returncode=None, return_code=None)
            return JSONResponse({"ok": False, "message": msg, "error": msg})
        RUN_PROC = proc
    set_run_state(running=True, started_at=now_iso(), current_step="starting", returncode=None, return_code=None, error=None)
    threading.Thread(target=monitor, args=(proc,), daemon=True).start()
    return JSONResponse({"ok": True, "message": f"Started pid={proc.pid}"})


@app.get("/api/v1/run/status")
def api_run_status():
    state = dict(RUN_STATE_MEM)
    state["running"] = bool(RUN_PROC is not None and RUN_PROC.poll() is None)
    state["log_tail"] = tail_log()
    return JSONResponse(state)


@app.post("/api/v1/run/stop")
def api_run_stop():
    with RUN_LOCK:
        proc = RUN_PROC
    if proc is None or proc.poll() is not None:
        return JSONResponse({"ok": False, "message": "No active run"})
    try:
        proc.terminate()
        return JSONResponse({"ok": True, "message": "Stop requested"})
    except Exception as e:
        return JSONResponse({"ok": False, "message": str(e)})
