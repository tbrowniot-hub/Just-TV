#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

TVMAZE_ROOT = "https://api.tvmaze.com"

def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def safe_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default

def parse_disc_value(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None

def minutes_to_seconds(m: float) -> int:
    return int(round(m * 60))

def seconds_to_minutes(s: int) -> float:
    return float(s) / 60.0

def tvmaze_show_year(show_id: int) -> Optional[int]:
    try:
        url = f"{TVMAZE_ROOT}/shows/{show_id}"
        req = Request(url, headers={"User-Agent": "DiscMapperTV/0.2"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        premiered = (data.get("premiered") or "").strip()  # "1997-08-13"
        if len(premiered) >= 4 and premiered[:4].isdigit():
            return int(premiered[:4])
    except Exception:
        return None
    return None

DEFAULT_CONFIG = {
    "makemkv_path": r"C:\\Program Files (x86)\\MakeMKV\\makemkvcon64.exe",
    "ffprobe_path": "ffprobe",
    "drive_index": 0,
    "auto_eject": True,

    "raw_root_tv": r"C:\\MediaPipeline\\Ripping_Staging\\1_Raw_Dumps\\TV",
    "ready_root_tv": r"C:\\MediaPipeline\\Ripping_Staging\\3_Ready_For_Unraid\\TV",
    "review_root_tv": r"C:\\MediaPipeline\\Ripping_Staging\\2_Work_Bench\\TV Review",
    "unable_root": r"C:\\MediaPipeline\\Ripping_Staging\\2_Work_Bench\\Unable_to_Read",
    "done_root_tv": r"C:\\MediaPipeline\\Ripping_Staging\\1_Raw_Dumps\\TV\\_done",

    # DIRTY MODE: rip all (but ignore tiny junk)
    "rip_floor_minutes": 6,
    "use_manifest_driven_minlength": True,
    "manifest_minlength_buffer_minutes": 2,
    "write_sidecar_json": True,

    # Matching: use "typical runtime" from disc + manifest as secondary guidance
    "match_manifest_buffer_minutes": 12,
    "match_typical_buffer_minutes": 8,
    "special_runtime_delta_minutes": 10,

    "max_avg_error_minutes_for_auto": 4.0,
    "skip_title_penalty_minutes": 2.0,

    # Plex folder style
    "include_show_year_in_folder": True
}

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        write_json(path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    cfg = read_json(path) or {}
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg)
    return merged

def build_tv_index(manifest_csv: Path) -> Dict[str, Any]:
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    episodes_by_key: Dict[str, List[Dict[str, Any]]] = {}
    ignored = 0
    year_cache: Dict[int, Optional[int]] = {}

    for r in rows:
        series = (r.get("Series") or "").strip()
        season = to_int(r.get("Season"))
        disc = parse_disc_value(r.get("Disc") or "")
        if not series or season is None or disc is None:
            ignored += 1
            continue

        ep_title = (r.get("Episode Title") or "").strip()
        sxxeyy = (r.get("SxxEyy") or "").strip()
        ep_no = to_int(r.get("Episode Number"))
        min_rt = to_int(r.get("Min run length"))
        max_rt = to_int(r.get("Max run length"))
        pkg_index = to_int(r.get("index"))
        upc = (r.get("Upc") or r.get("UPC") or "").strip()
        imdb_url = (r.get("IMDb Url") or r.get("IMDB Url") or r.get("IMDb URL") or "").strip()
        physical_title = (r.get("Phyisical title") or r.get("Physical title") or "").strip()

        show_id = to_int(r.get("TVMaze Show ID"))
        show_year = None
        if show_id is not None:
            if show_id not in year_cache:
                year_cache[show_id] = tvmaze_show_year(show_id)
            show_year = year_cache[show_id]

        key = f"{series}||S{int(season):02d}||D{int(disc):02d}"
        episodes_by_key.setdefault(key, []).append({
            "series": series,
            "season": int(season),
            "disc": int(disc),
            "show_year": show_year,
            "sxxeyy": sxxeyy,
            "ep_no": ep_no,
            "episode_title": ep_title,
            "min_minutes": min_rt,
            "max_minutes": max_rt,
            "index": pkg_index,
            "upc": upc,
            "imdb_url": imdb_url,
            "physical_title": physical_title
        })

    for k, eps in episodes_by_key.items():
        eps.sort(key=lambda e: (e["ep_no"] is None, e["ep_no"] if e["ep_no"] is not None else 9999))

    discs = []
    for k, eps in episodes_by_key.items():
        discs.append({
            "key": k,
            "series": eps[0]["series"],
            "season": eps[0]["season"],
            "disc": eps[0]["disc"],
            "show_year": eps[0].get("show_year"),
            "episode_count": len(eps)
        })
    discs.sort(key=lambda d: (d["series"].lower(), d["season"], d["disc"]))

    return {
        "built_at": now_stamp(),
        "manifest_csv": str(manifest_csv),
        "ignored_rows_missing_series_season_or_disc": ignored,
        "discs": discs,
        "episodes_by_key": episodes_by_key
    }

def powershell_json(cmd: str) -> Any:
    ps = ["powershell", "-NoProfile", "-Command", cmd]
    out = subprocess.check_output(ps, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except Exception:
        return out

def get_cdrom_drives() -> List[Dict[str, Any]]:
    data = powershell_json("Get-CimInstance Win32_CDROMDrive | Select-Object Drive, MediaLoaded | ConvertTo-Json")
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    return []

def wait_for_disc_insert(poll_seconds: float = 2.0) -> str:
    while True:
        for d in get_cdrom_drives():
            if str(d.get("MediaLoaded")).lower() == "true":
                drv = d.get("Drive")
                if drv:
                    return str(drv)
        time.sleep(poll_seconds)

def eject_drive(drive_letter: str) -> None:
    dl = drive_letter.strip().upper()
    if not dl.endswith(":"):
        dl += ":"
    cmd = f"(New-Object -COM Shell.Application).NameSpace(17).ParseName('{dl}').InvokeVerb('Eject')"
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=False)

TITLE_NUM_RE = re.compile(r"(?:title|t)(\d{1,3})", re.IGNORECASE)

def file_title_index(p: Path) -> Optional[int]:
    m = TITLE_NUM_RE.search(p.name)
    return int(m.group(1)) if m else None

def makemkv_rip_all(makemkv_path: str, drive_index: int, out_dir: Path, minlength_seconds: int, log_path: Path) -> int:
    ensure_dir(out_dir)
    cmd = [makemkv_path, "-r", "mkv", f"disc:{drive_index}", "all", str(out_dir), f"--minlength={minlength_seconds}"]
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return int(p.returncode)

def ffprobe_duration_seconds(ffprobe_path: str, file_path: Path) -> Optional[int]:
    cmd = [
        ffprobe_path, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace").strip()
        if not out:
            return None
        return int(round(float(out)))
    except Exception:
        return None

def compute_typical_runtime_seconds(files: List[Dict[str, Any]], episode_count: int) -> Optional[int]:
    # Robust typical runtime estimator: avoid being biased toward the shortest titles (often junk).
    durs = sorted([f["duration_s"] for f in files if isinstance(f.get("duration_s"), int)])
    if not durs:
        return None

    # Trim extremes (bottom/top 20%) when we have enough samples.
    if len(durs) >= 5:
        trim = max(1, int(round(len(durs) * 0.20)))
        core = durs[trim:len(durs) - trim]
        if core:
            durs = core

    try:
        return int(statistics.median(durs))
    except Exception:
        return int(durs[len(durs) // 2])

def build_episode_windows(
    eps: List[Dict[str, Any]],
    typical_s: int,
    manifest_buf_min: int,
    typical_buf_min: int,
    special_delta_min: int
) -> List[Dict[str, Any]]:
    out = []
    typical_min_s = max(60, typical_s - minutes_to_seconds(typical_buf_min))
    typical_max_s = typical_s + minutes_to_seconds(typical_buf_min)
    typical_m = seconds_to_minutes(typical_s)

    for e in eps:
        mi = to_int(e.get("min_minutes"))
        ma = to_int(e.get("max_minutes"))

        if mi is None or ma is None:
            out.append({**e, "min_s": typical_min_s, "max_s": typical_max_s})
            continue

        expected_mid_m = (mi + ma) / 2.0
        if abs(expected_mid_m - typical_m) >= special_delta_min:
            raw_min_s = max(60, minutes_to_seconds(max(1, mi - manifest_buf_min)))
            raw_max_s = minutes_to_seconds(ma + manifest_buf_min)
            out.append({**e, "min_s": raw_min_s, "max_s": raw_max_s})
            continue

        raw_min_s = max(60, minutes_to_seconds(max(1, mi - manifest_buf_min)))
        raw_max_s = minutes_to_seconds(ma + manifest_buf_min)

        if raw_min_s <= typical_s <= raw_max_s:
            final_min = max(raw_min_s, typical_min_s)
            final_max = min(raw_max_s, typical_max_s)
        else:
            final_min, final_max = typical_min_s, typical_max_s

        out.append({**e, "min_s": final_min, "max_s": final_max})

    return out

def dp_map_files_to_episodes(
    eps_win: List[Dict[str, Any]],
    files: List[Dict[str, Any]],
    typical_s: int,
    skip_penalty_minutes: float
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    m, n = len(eps_win), len(files)
    INF = 10**18
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    take = [[0] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = 0.0
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + skip_penalty_minutes

    def cost(i: int, j: int) -> float:
        e = eps_win[i - 1]
        f = files[j - 1]
        dur = f.get("duration_s")
        if dur is None:
            return float("inf")
        if not (e["min_s"] <= dur <= e["max_s"]):
            return float("inf")
        mid = (e["min_s"] + e["max_s"]) / 2.0
        return abs(dur - mid) / 60.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = dp[i][j - 1] + skip_penalty_minutes
            t = 0
            c = cost(i, j)
            if c != float("inf"):
                cand = dp[i - 1][j - 1] + c
                if cand < best:
                    best = cand
                    t = 1
            dp[i][j] = best
            take[i][j] = t

    if dp[m][n] >= INF / 2:
        return None, float("inf")

    pairs: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if take[i][j] == 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        else:
            j -= 1

    if i != 0:
        return None, float("inf")

    pairs.reverse()
    errs = [abs(files[fj]["duration_s"] - ((eps_win[ei]["min_s"] + eps_win[ei]["max_s"]) / 2.0)) / 60.0 for ei, fj in pairs]
    avg_err = sum(errs) / len(errs) if errs else float("inf")
    return pairs, avg_err

def show_folder_name(series: str, year: Optional[int], include_year: bool) -> str:
    s = safe_filename(series)
    if include_year and year:
        return f"{s} ({year})"
    return s

def tv_dest_paths(ready_root: Path, series: str, show_year: Optional[int], include_year: bool, season: int, sxxeyy: str, ep_title: str, pkg_index: Optional[int] = None) -> Tuple[Path, Path]:
    series_clean = safe_filename(series)
    show_folder = show_folder_name(series, show_year, include_year)
    season_folder = f"Season {season:02d}"
    dest_dir = ready_root / show_folder / season_folder

    sxx = sxxeyy.strip() if sxxeyy else f"S{season:02d}E??"
    ep_safe = safe_filename(ep_title)

    fname = f"{series_clean} - {sxx}"
    if ep_safe:
        fname += f" - {ep_safe}"
    if pkg_index is not None:
        fname += f" [CLZ_{int(pkg_index)}]"
    fname += ".mkv"
    return dest_dir, dest_dir / fname

def cmd_import_manifest(args: argparse.Namespace) -> None:
    idx = build_tv_index(Path(args.manifest).expanduser())
    write_json(Path(args.out).expanduser(), idx)
    print(f"[DiscMapper TV] Wrote index: {args.out}")
    print(f"[DiscMapper TV] Discs indexed: {len(idx.get('discs', []))}")
    print(f"[DiscMapper TV] Ignored rows missing Series/Season/Disc: {idx.get('ignored_rows_missing_series_season_or_disc')}")

def cmd_queue_builder(args: argparse.Namespace) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    index_path = Path(args.index).expanduser()
    out_path = Path(args.out).expanduser()
    idx = read_json(index_path)
    discs = idx.get("discs", [])
    if not discs:
        print("No discs found in index. Fill Disc column in tv_manifest.csv and re-import.")
        return

    def label(d: Dict[str, Any]) -> str:
        return f"{d['series']} | S{int(d['season']):02d} | D{int(d['disc']):02d} | {d.get('episode_count',0)} eps"

    root = tk.Tk()
    root.title("DiscMapper TV v0.2 â€” Build Rip Queue (Discs)")
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(1, weight=1)

    tk.Label(root, text="Search (series):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
    search = tk.StringVar()
    ent = tk.Entry(root, textvariable=search)
    ent.grid(row=0, column=0, sticky="ew", padx=120, pady=6)

    lf = ttk.LabelFrame(root, text="Available discs"); lf.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
    rf = ttk.LabelFrame(root, text="Rip queue (top â†’ bottom)"); rf.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
    lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
    rf.rowconfigure(0, weight=1); rf.columnconfigure(0, weight=1)

    left = tk.Listbox(lf, exportselection=False)
    right = tk.Listbox(rf, exportselection=False)
    left.grid(row=0, column=0, sticky="nsew")
    right.grid(row=0, column=0, sticky="nsew")

    def refresh():
        q = search.get().strip().lower()
        left.delete(0, tk.END)
        for d in discs:
            if q and q not in d["series"].lower():
                continue
            left.insert(tk.END, label(d))

    def disc_from_left_idx(i: int) -> Dict[str, Any]:
        q = search.get().strip().lower()
        filtered = [d for d in discs if (not q or q in d["series"].lower())]
        return filtered[i]

    def add():
        sel = left.curselection()
        if not sel:
            return
        d = disc_from_left_idx(sel[0])
        right.insert(tk.END, label(d))

    def remove():
        sel = right.curselection()
        if not sel:
            return
        right.delete(sel[0])

    def up():
        sel = right.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        t = right.get(i)
        right.delete(i)
        right.insert(i - 1, t)
        right.selection_clear(0, tk.END)
        right.selection_set(i - 1)

    def down():
        sel = right.curselection()
        if not sel or sel[0] >= right.size() - 1:
            return
        i = sel[0]
        t = right.get(i)
        right.delete(i)
        right.insert(i + 1, t)
        right.selection_clear(0, tk.END)
        right.selection_set(i + 1)

    def save():
        map_lbl = {label(d): d["key"] for d in discs}
        keys = []
        for i in range(right.size()):
            k = map_lbl.get(right.get(i))
            if k:
                keys.append(k)
        if not keys:
            messagebox.showwarning("Queue", "Queue is empty.")
            return
        write_json(out_path, {"built_at": now_stamp(), "index": str(index_path), "queue_keys": keys})
        messagebox.showinfo("Saved", f"Saved queue:\n{out_path}")

    bf = tk.Frame(root); bf.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
    ttk.Button(bf, text="Add â†’", command=add).pack(side="left", padx=4)
    ttk.Button(bf, text="Remove", command=remove).pack(side="left", padx=4)
    ttk.Button(bf, text="Up", command=up).pack(side="left", padx=4)
    ttk.Button(bf, text="Down", command=down).pack(side="left", padx=4)
    ttk.Button(bf, text="Save Queue", command=save).pack(side="left", padx=8)
    ttk.Button(bf, text="Close", command=root.destroy).pack(side="left", padx=4)

    search.trace_add("write", lambda *_: refresh())
    refresh()
    root.mainloop()

def cmd_rip_queue(args: argparse.Namespace) -> None:
    index = read_json(Path(args.index).expanduser())
    queue = read_json(Path(args.queue).expanduser())
    cfg = load_config(Path(args.config).expanduser())

    makemkv = cfg["makemkv_path"]
    ffprobe = cfg["ffprobe_path"]
    drive_index = int(cfg["drive_index"])
    auto_eject = bool(cfg["auto_eject"])

    raw_root = Path(cfg["raw_root_tv"])
    ready_root = Path(cfg["ready_root_tv"])
    review_root = Path(cfg["review_root_tv"])
    unable_root = Path(cfg["unable_root"])
    done_root = Path(cfg["done_root_tv"])

    ensure_dir(raw_root); ensure_dir(ready_root); ensure_dir(review_root); ensure_dir(unable_root); ensure_dir(done_root)

    floor_min = int(cfg["rip_floor_minutes"])
    floor_s = minutes_to_seconds(floor_min)

    manifest_buf = int(cfg["match_manifest_buffer_minutes"])
    typical_buf = int(cfg["match_typical_buffer_minutes"])
    special_delta = int(cfg["special_runtime_delta_minutes"])
    skip_penalty = float(cfg["skip_title_penalty_minutes"])
    max_avg_err = float(cfg["max_avg_error_minutes_for_auto"])
    include_year = bool(cfg["include_show_year_in_folder"])

    episodes_by_key = index.get("episodes_by_key", {})
    keys = queue.get("queue_keys", [])
    if not keys:
        print("[DiscMapper TV] Queue empty.")
        return

    total = len(keys)
    for pos, key in enumerate(keys, start=1):
        eps = episodes_by_key.get(key)
        if not eps:
            print(f"[DiscMapper TV] Missing key in index: {key}")
            continue

        # Determine minlength for ripping/filtering: manifest-driven when possible so we don't drop short episodes.
        use_manifest_minlen = bool(cfg.get("use_manifest_driven_minlength", True))
        minlen_buf = int(cfg.get("manifest_minlength_buffer_minutes", 2))
        disc_floor_min = floor_min
        if use_manifest_minlen:
            mins = [to_int(e.get("min_minutes")) for e in eps if to_int(e.get("min_minutes")) is not None]
            if mins:
                disc_floor_min = max(1, int(min(mins)) - max(0, minlen_buf))
        disc_floor_s = minutes_to_seconds(disc_floor_min)

        series = eps[0]["series"]
        season = int(eps[0]["season"])
        disc = int(eps[0]["disc"])
        show_year = eps[0].get("show_year")

        print("\n" + "="*80)
        print(f"[{pos}/{total}] NEXT DISC: {series} â€” Season {season:02d} Disc {disc:02d} ({len(eps)} eps)")
        print("Insert disc now... (Ctrl+C to stop)")
        drive_letter = wait_for_disc_insert()
        print(f"[DiscMapper TV] Disc detected: {drive_letter}")

        job_name = safe_filename(f"{series} - S{season:02d}D{disc:02d} - {now_stamp()}")
        job_dir = raw_root / job_name
        ensure_dir(job_dir)

        rip_log = job_dir / f"makemkv_rip_{now_stamp()}.log"
        print(f"[DiscMapper TV] Dirty mode: ripping ALL titles. minlength={disc_floor_s}s")
        rc = makemkv_rip_all(makemkv, drive_index, job_dir, disc_floor_s, rip_log)
        if rc != 0:
            print(f"[DiscMapper TV] WARNING: MakeMKV returned exit {rc}. Will continue if any MKVs were produced.")

        if auto_eject:
            try:
                eject_drive(drive_letter)
            except Exception:
                print("[DiscMapper TV] WARNING: auto-eject failed")

        mkvs = sorted([p for p in job_dir.rglob("*.mkv") if p.is_file()])
        if not mkvs:
            print("[DiscMapper TV] No MKVs produced â†’ Unable_to_Read")
            shutil.move(str(job_dir), str(unable_root / job_name))
            continue

        files: List[Dict[str, Any]] = []
        for p in mkvs:
            dur = ffprobe_duration_seconds(ffprobe, p)
            if dur is None:
                print("[DiscMapper TV] ffprobe failed (install FFmpeg) â†’ Review")
                shutil.move(str(job_dir), str(review_root / job_name))
                break
            if dur < disc_floor_s:
                continue
            files.append({
                "path": str(p),
                "name": p.name,
                "duration_s": dur,
                "title_index": file_title_index(p),
                "size": p.stat().st_size
            })
        else:
            if len(files) < len(eps):
                print(f"[DiscMapper TV] Not enough episode-length files ({len(files)} found for {len(eps)} eps; minlength={disc_floor_min}m) â†’ Review")
                shutil.move(str(job_dir), str(review_root / job_name))
                continue

            files.sort(key=lambda f: (f["title_index"] is None, f["title_index"] or 9999, f["name"].lower()))
            typical_s = compute_typical_runtime_seconds(files, episode_count=len(eps))
            if typical_s is None:
                print("[DiscMapper TV] Could not compute typical runtime â†’ Review")
                shutil.move(str(job_dir), str(review_root / job_name))
                continue

            eps_win = build_episode_windows(
                eps, typical_s=typical_s,
                manifest_buf_min=manifest_buf,
                typical_buf_min=typical_buf,
                special_delta_min=special_delta
            )

            pairs, avg_err = dp_map_files_to_episodes(eps_win, files, typical_s=typical_s, skip_penalty_minutes=skip_penalty)
            if pairs is None or avg_err > max_avg_err or len(pairs) != len(eps_win):
                print(f"[DiscMapper TV] Mapping uncertain (avg_err={avg_err:.2f} min) â†’ Review")
                shutil.move(str(job_dir), str(review_root / job_name))
                continue

            moved = 0
            for ei, fj in pairs:
                e = eps_win[ei]
                f = files[fj]
                src = Path(f["path"])

                dest_dir, dest_file = tv_dest_paths(
                    ready_root,
                    series=e["series"],
                    show_year=show_year,
                    include_year=include_year,
                    season=int(e["season"]),
                    sxxeyy=e.get("sxxeyy") or f"S{season:02d}E{ei+1:02d}",
                    ep_title=e.get("episode_title") or "",
                    pkg_index=to_int(e.get("index"))
                )
                ensure_dir(dest_dir)

                final = dest_file
                if final.exists():
                    final = final.with_name(final.stem + f"__dup_{now_stamp()}.mkv")

                shutil.move(str(src), str(final))
                moved += 1

                if bool(cfg.get("write_sidecar_json", True)):
                    sidecar = final.with_name(final.stem + ".discmapper.json")
                    meta = {
                        "series": e.get("series"),
                        "season": e.get("season"),
                        "disc": e.get("disc"),
                        "sxxeyy": e.get("sxxeyy"),
                        "episode_title": e.get("episode_title"),
                        "index": to_int(e.get("index")),
                        "upc": e.get("upc"),
                        "imdb_url": e.get("imdb_url"),
                        "physical_title": e.get("physical_title"),
                        "source_title_index": f.get("title_index"),
                        "source_filename": f.get("name"),
                        "duration_s": f.get("duration_s"),
                        "bytes": f.get("size"),
                        "ripped_job_dir": str(job_dir),
                        "final_path": str(final),
                        "mapped_at": now_stamp(),
                    }
                    try:
                        write_json(sidecar, meta)
                    except Exception:
                        pass

            # Move job folder to done (keeps any leftovers/logs for audit)
            if job_dir.exists():
                shutil.move(str(job_dir), str(done_root / job_name))

            print(f"[DiscMapper TV] SUCCESS: moved {moved} eps â†’ {ready_root}\\{show_folder_name(series, show_year, include_year)}\\Season {season:02d}")

def main() -> None:
    p = argparse.ArgumentParser(prog="discmapper_tv_v02")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("import-manifest")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.set_defaults(func=cmd_import_manifest)

    b = sub.add_parser("queue-builder")
    b.add_argument("--index", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_queue_builder)

    c = sub.add_parser("rip-queue")
    c.add_argument("--index", required=True)
    c.add_argument("--queue", required=True)
    c.add_argument("--config", required=True)
    c.set_defaults(func=cmd_rip_queue)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
