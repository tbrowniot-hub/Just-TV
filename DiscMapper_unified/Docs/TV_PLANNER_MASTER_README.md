# TV Planner Master README (Behavior Freeze)

This document freezes the **current** TV Planning UI behavior before any UI refactor.

Scope:
- Intended for future developers
- Intended for future Codex runs
- Intended as a personal memory aid for full workflow and edge cases

Important:
- This describes the UI/planning layer behavior currently implemented in `tv_ui_app.py`.
- This is a documentation snapshot, not a design proposal.
- Export compatibility with the TV rip engine is treated as non-negotiable.

---

## 1) High Level Purpose

The TV Planning UI is a planning/orchestration layer that sits in front of the DiscMapper TV rip engine.

Pipeline concept:
1. CLZ CSV is imported (physical collection metadata).
2. Planner organizes packages (including parent/child box-set grouping).
3. Package-level mapping links package -> TV show -> season, then maps **disc -> episodes**.
4. Queue Builder builds an ordered rip queue from mapped discs.
5. Run page launches the TV ripper against index + queue + config.

Functional intent:
- Separate metadata/planning work from ripping work.
- Allow iterative mapping edits before queue build.
- Make queue explicit and reorderable.
- Preserve engine-compatible artifacts (`tv_manifest.csv`, `tv_index.json`, `tv_queue.json`).

In short:
- UI = planning + mapping + queue authoring + run control.
- Engine (`discmapper_tv_v02.py`/`discmapper_tv_v02_1.py`) = actual rip execution.

---

## 2) Directory Structure

### Canonical project shape (expected)

```text
DiscMapper_unified/
  Inputs/
  Data/
    Indexes/
    Queues/
  App/
  Docs/
```

### Current implementation note (important)

The current `tv_ui_app.py` in this repo uses **repo-root local files** for the core artifacts:
- `tv_manifest.csv`
- `tv_index.json`
- `tv_queue.json`

And uses `Data/` for UI state:
- `Data/tv_ui.db`
- `Data/Config/app_config.json`
- `Data/Logs/tv_run.log`
- `Data/Logs/tv_run_state.json`

CLZ default path behavior:
- configurable: `clz_export_path` in app config
- fallback: `CLZ_export.csv` in repo root

### Generated files and consumers

- `tv_manifest.csv`
  - Input manifest for index build.
  - Consumed by `build_tv_index(...)` in TV engine module.

- `tv_index.json`
  - Built from manifest by engine helper.
  - Referenced by queue payload and rip command.

- `tv_queue.json`
  - Queue payload with queue keys/order.
  - Consumed by rip-queue command in engine.

Engine entry used by Run page:
- `discmapper_tv_v02_1.py rip-queue --index ... --queue ... --config ...`

---

## 3) End-to-End Workflow (Exact Current Behavior)

## Step 1 — PLAN PAGE (`/plan`)

Purpose:
- Ingest CLZ data
- review TV package rows
- maintain parent/child package relations
- perform initial artifact commands
- navigate into package mapping

### Buttons and what each does

- **Import/Refresh CLZ**
  - Calls `POST /plan/import-clz`
  - Reads selected CLZ CSV
  - Upserts rows into `clz_item`
  - Header mapping is case-insensitive
  - Stores raw CLZ row JSON in DB (`raw_json`)
  - Updates `has_is_tv_column` flag in app config

- **Choose CLZ CSV...**
  - Calls `POST /plan/pick-clz`
  - Uses native file picker (`tkinter.filedialog.askopenfilename`)
  - Saves selected path to config key `clz_export_path`
  - Fallback message if picker unavailable

- **Build tv_index.json now**
  - Calls `POST /plan/build-index`
  - Requires `tv_manifest.csv` to exist
  - Imports engine module and runs `build_tv_index(TV_MANIFEST)`
  - Writes `tv_index.json`
  - Returns guidance message to click Title for mapping

- **Export tv_manifest.csv**
  - Calls `POST /plan/export-manifest`
  - Current behavior: does not generate manifest in this repo
  - If manifest already exists, indicates present
  - Otherwise returns informational message to provide it

- **Export tv_queue.json**
  - Calls `POST /plan/export-queue`
  - Builds queue from currently mapped rows (`map_rows(...)`) with `episode_count > 0`
  - Forces include + sequential priority
  - Writes queue using engine-compatible schema

- **Validate**
  - `GET /plan/validate`
  - Returns JSON with existence checks for manifest/index/queue and CLZ path state

### CLZ import internals

Input resolution:
1. `clz_export_path` if configured and exists
2. fallback root `CLZ_export.csv`
3. else import unavailable

Field detection (case-insensitive):
- Title
- Index / CLZ Index / Nr / Number
- Barcode / UPC
- Box Set
- Is TV Series
- Nr Discs / Discs / Disc Count
- Format
- IMDb Url
- Notes

Behavior:
- `clz_index` is taken from mapped index field or row number fallback
- `barcode` stored as text (string-safe handling)
- `is_tv_series` normalized from truthy tokens (`1,true,yes,y,t`)

### Parent/child behavior

On the plan listing:
- Manual parent set/clear via:
  - `POST /plan/parent/{clz_index}`
  - `POST /plan/parent-clear/{clz_index}`
- Auto inference also applied for display/grouping:
  - if row has `box_set` and no explicit parent
  - tries title match to another row title (case-insensitive)
  - grouped under inferred parent when grouping enabled

### Filters

- **TV only**
  - If `Is TV Series` column detected, uses `is_tv_series == 1`.
  - Else fallback title heuristic matches:
    - `season|series|complete|volume|vol|book|disc`

- **Group box sets**
  - Controls whether parent-child nesting is rendered.

### Critical user transition

Primary next step from Plan:
- Click package **Title** (or Map link) -> `/package/{clz_index}`

---

## Step 2 — PACKAGE DETAIL / MAPPING (`/package/{clz_index}`)

Purpose:
- Bind CLZ package to TV show + season
- Populate episode cache from TVMaze
- Define discs
- Assign episodes per disc

### Package metadata + parent controls

Shows:
- title, barcode, format, disc count, box set, notes

Controls:
- Set parent
- Clear parent

### Link to Show (TVMaze)

Search:
- `GET /package/{clz_index}?q=<text>`
- Uses TVMaze search endpoint:
  - `https://api.tvmaze.com/search/shows?q=...`

Link:
- `POST /package/{clz_index}/show-link`
- Persists to `package_link`:
  - `tvmaze_show_id`
  - `show_name`
  - optional imdb id

### Refresh Episodes

- `POST /package/{clz_index}/episodes-refresh`
- Requires linked `tvmaze_show_id`
- Calls TVMaze episodes endpoint:
  - `https://api.tvmaze.com/shows/{id}/episodes`
- Upserts all returned episodes into `episode_cache`:
  - season, episode_number, SxxEyy code, title, TVMaze episode id, runtime, payload JSON

### Season selection

- `POST /package/{clz_index}/season`
- Persists `selected_season` in `package_link`

### Disc creation model

Disc reset:
- `POST /package/{clz_index}/discs-reset`
- Ensures discs 1..N exist

Manual add/update:
- `POST /package/{clz_index}/disc-save`
- upserts single disc number + optional label

Disc UID format:
- `CLZ{clz_index}-DNN`
  - Examples:
    - `CLZ123-D01`
    - `CLZ42-D03`

### Episode assignment model

Current UX split:
- Package page shows disc summaries only.
- Assignment happens on disc page:
  - `/disc/{disc_uid}`

Storage:
- table `disc_episode_map`
- one row per `(disc_uid, episode_code)`

---

## Step 2b — DISC ASSIGNMENT PAGE (`/disc/{disc_uid}`)

Purpose:
- Full checklist assignment for one disc against selected show season.

Loads:
- disc metadata (`disc`)
- package row (`clz_item`)
- package show link (`package_link`)
- selected season
- cached episodes for show+season (`episode_cache`)
- existing disc mapping (`disc_episode_map`)

Table columns:
- checkbox
- `SxxEyy`
- episode title
- runtime minutes

Convenience:
- Save button at top and bottom
- Select all
- Clear all

Save behavior:
- `POST /disc/{disc_uid}/save`
- Deletes existing rows in `disc_episode_map` for that disc
- Inserts checked episodes back
- Ordering follows episode number ascending (via cached episode lookup)
- Redirects to package page with success message

---

## Step 3 — MAP PAGE (`/map`) [Summary-oriented]

Purpose:
- View/save include + priority over mapped candidate rows.

What `map_rows(...)` currently represents:
- Built from mapped discs in SQLite (`disc`, `disc_episode_map`, `package_link`, `clz_item`)
- Generates candidate queue row fields:
  - `queue_key`
  - series/season/disc
  - episode count
  - short count (runtime threshold check)

Include/priority:
- persisted in `mapped_item`
- can be edited and saved from `/map`

Min-clip behavior in summary:
- uses `min_clip_seconds` from app config
- short episode counts are reported; rows not dropped from source mapping

---

## Step 4 — QUEUE PAGE (`/queue`)

Purpose:
- Build actual rip queue from mapped disc candidates.

Model:
- Two panes
  - Candidates
  - Queue draft

Draft persistence:
- `queue_draft` table stores included/order state.

Operations:
- Add selected candidates
- Save draft
- Move up/down
- Remove item
- Clear draft

Build queue:
- `POST /queue/build`
- Writes `tv_queue.json` schema:
  - `built_at`
  - `index` (path to `tv_index.json`)
  - `queue_keys` (ordered list)

Candidate source of truth:
- SQLite mapped discs (`disc` + `disc_episode_map`) transformed through `map_rows(...)`
- Not directly from CLZ alone

---

## Step 5 — RUN PAGE (`/run`)

Purpose:
- Validate preconditions and start/monitor TV rip run.

Preconditions enforced on start:
- queue file exists
- queue has > 0 items
- config roots exist (`rips_staging_root`, `rips_complete_root`)
- engine entrypoint exists (`discmapper_tv_v02_1.py`)
- index file exists

Run start:
- `POST /api/v1/run/start`
- Executes:
  - `python discmapper_tv_v02_1.py rip-queue --index tv_index.json --queue tv_queue.json --config config.json`

Status:
- `GET /api/v1/run/status`
- Includes running state + log tail

Stop:
- `POST /api/v1/run/stop` (best effort terminate)

Logs/state:
- `Data/Logs/tv_run.log`
- `Data/Logs/tv_run_state.json`

---

## 4) Data Model

Core conceptual entities:
- **Package** (`clz_item`)
  - physical package from CLZ export
- **Show Link** (`package_link`)
  - package -> TVMaze show + selected season
- **Disc** (`disc`)
  - package has one or more discs
- **Episode Cache** (`episode_cache`)
  - authoritative season episode metadata fetched from TVMaze
- **Disc Episode Assignment** (`disc_episode_map`)
  - many-to-many flattened as rows per disc + episode code

### Key IDs

- `clz_index`
  - package primary identifier from CLZ
- `parent_index`
  - package grouping pointer for box sets
- `disc_uid`
  - stable disc ID: `CLZ{clz_index}-DNN`
- `episode_code`
  - `SxxEyy`

### Relationship graph

```text
clz_item (package)
  -> package_link (optional show binding)
  -> disc (1..N)
      -> disc_episode_map (0..M episode assignments)

package_link.tvmaze_show_id
  -> episode_cache (show seasons/episodes)
```

### Queue identity model

Queue key currently built as:
- `{show_name}||S{season:02d}||D{disc:02d}`

This is what queue draft and `tv_queue.json` carry in `queue_keys`.

---

## 5) Generated Artifacts (Production + Consumption)

## `tv_manifest.csv`

Produced:
- Not generated by this UI currently.
- UI can trigger index build only if this file already exists.

Consumed:
- by engine helper `build_tv_index(...)` during `/plan/build-index`.

## `tv_index.json`

Produced:
- `/plan/build-index`
- generated by TV engine module from manifest

Consumed:
- referenced in queue payload (`index` field)
- used by rip engine invocation

## `tv_queue.json`

Produced:
- `/plan/export-queue` (quick export path)
- `/queue/build` (draft-based, preferred queue UX path)

Schema (must stay engine-compatible):
- `built_at: str`
- `index: str` (index path)
- `queue_keys: list[str]`

Consumed:
- rip engine `rip-queue` command

---

## 6) Known UX Problems (Current Pain Points)

These issues are intentionally documented to justify refactor work while preserving functionality.

- Plan page has many actions with mixed abstraction levels:
  - CLZ import, index build, queue export, validate all together
- Labels are developer-facing:
  - `tv_index.json`, `tv_queue.json`, manifest phrasing
- Workflow guidance is present but still easy to bypass:
  - users may click queue/run before package mapping complete
- Separation between Plan, Package, Summary(Map), Queue can be confusing:
  - state is persisted but mental model is not explicit for first-time users
- Engine/technical terms surface in user-facing flow:
  - index path, queue keys, debug details
- CLZ/TVMaze dependency order can cause empty states:
  - no show link -> no episode cache -> no disc mapping -> no queue candidates

---

## 7) Future Refactor Goals (Without Losing Features)

Refactor objective:
- Simplify navigation and guidance while preserving all current capabilities and data semantics.

Must preserve:
- CLZ import + parent/child grouping controls
- package show linking + episode refresh + season select
- disc creation + per-disc episode assignment
- persisted queue draft and ordered queue build
- run preflight checks + log/state monitoring
- engine artifact compatibility (`manifest/index/queue` expectations)

Suggested UX direction:
- More wizard-like next-step guidance
- Fewer competing actions on Plan
- clearer distinction between:
  - metadata preparation
  - mapping completion
  - queue publication
  - execution

Non-goals:
- Do not alter rip engine contract.
- Do not remove mapping fidelity or persistence depth.

---

## Appendix: Route Inventory (Current)

Plan:
- `GET /plan`
- `POST /plan/import-clz`
- `POST /plan/pick-clz`
- `POST /plan/build-index`
- `POST /plan/export-manifest`
- `POST /plan/export-queue`
- `GET /plan/validate`
- `POST /plan/parent/{clz_index}`
- `POST /plan/parent-clear/{clz_index}`

Package + Disc mapping:
- `GET /package/{clz_index}`
- `POST /package/{clz_index}/show-link`
- `POST /package/{clz_index}/parent`
- `POST /package/{clz_index}/parent-clear`
- `POST /package/{clz_index}/season`
- `POST /package/{clz_index}/episodes-refresh`
- `POST /package/{clz_index}/discs-reset`
- `POST /package/{clz_index}/disc-save`
- `GET /disc/{disc_uid}`
- `POST /disc/{disc_uid}/save`

Summary:
- `GET /map`
- `POST /map/save`

Queue:
- `GET /queue`
- `POST /queue/add`
- `POST /queue/save-draft`
- `POST /queue/remove`
- `POST /queue/move-up`
- `POST /queue/move-down`
- `POST /queue/clear`
- `POST /queue/build`

Config:
- `GET /config`
- `GET /api/v1/config`
- `POST /api/v1/config/validate`
- `POST /api/v1/config/save`
- `POST /api/v1/config/browse`

Run:
- `GET /run`
- `POST /api/v1/run/start`
- `GET /api/v1/run/status`
- `POST /api/v1/run/stop`

