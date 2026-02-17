# Just-TV
DiscMapper TV queue ripper.

## Web UI (plan -> map -> queue -> run)
Start UI:
`.\run_tv_ui.ps1`
Open:
`http://127.0.0.1:8787`

Flow:
1. `Plan` page: CLZ planning table workflow (import CLZ, parent override, build index, export actions, validate).
2. `Map` page: review mapped discs from `tv_index.json`, save include/priority, use Show short clips toggle.
3. `Queue` page: build `tv_queue.json` (engine schema unchanged: `built_at`, `index`, `queue_keys`).
4. `Run` page: press Start to launch ripper and watch status/log tail.

Persistent app config:
- `Data/Config/app_config.json`
- Keys: `rips_staging_root`, `rips_complete_root`, `min_clip_seconds`
- Saved values are synced into engine `config.json`.

Min length behavior:
- Canonical threshold is `min_clip_seconds` (seconds).
- UI marks `too_short` and hides short clips by default (`Show short clips` toggle).
- Engine allows short clips on manifest-explicit mapped discs.

## Smoke checklist
1. Run `.\run_tv_ui.ps1` and open `http://127.0.0.1:8787`.
2. `/plan`: Import/Refresh CLZ, confirm table + TV count banner + parent override actions.
3. `/plan`: Build `tv_index.json` (requires `tv_manifest.csv` present).
4. `/map`: verify short clips are hidden by default and appear with Show short clips.
5. `/queue`: build queue and confirm `tv_queue.json` includes `queue_keys`.
6. `/run`: start/stop run and confirm live status + log tail updates.

## Canonical run entrypoint
Run `run_rip_queue.ps1` from this repo root. It invokes `discmapper_tv_v02_1.py rip-queue` with `tv_index.json`, `tv_queue.json`, and `config.json`.

## Output contract
- Raw disc jobs are created under `rips_staging_root` (default: `C:\MediaPipeline\Ripping_Staging\1_Raw_Dumps\TV`).
- Successful mapped episodes are moved to `final_tv_root` (default: `C:\MediaPipeline\_QUEUE\TV`) as:
  `Show Name (Year optional)\Season XX\...`
- Success writes/overwrites `_READY.txt` in each touched season folder.
- Review/Unable never write to `final_tv_root` and never create `_READY.txt`.

## Review and Unable routing
- Reports are always written to:
  `C:\MediaPipeline\Ripping_Staging\2_Work_Bench\TV Review\<jobfolder>\match_report.json|csv`
- Leftovers are routed to:
  `...\TV Review\<jobfolder>\Leftovers\`
- Review keeps raw job capture under:
  `...\TV Review\<jobfolder>\raw\`
- If no MKVs are produced, the job is moved to:
  `C:\MediaPipeline\Ripping_Staging\2_Work_Bench\Unable_to_Read\<jobfolder>\`
  while still writing a report in Review.
