# Just-TV
DiscMapper TV queue ripper.

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
