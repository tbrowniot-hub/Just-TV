# DiscMapper v0.2 — TV Shows (Windows)

Final files land here (auto-created):
C:\MediaPipeline\Ripping_Staging\3_Ready_For_Unraid\TV\<Series>\Season 01\...

Your manifest (truth) must be: tv_manifest.csv with Disc filled (you map Disc from the box/back cover).

## Steps

1) Build tv_index.json
.\run_import_manifest.ps1 -ManifestCsv "C:\Users\Morta\Documents\tv_manifest_out\tv_manifest.csv"

2) Build queue (GUI)
.\run_queue_builder.ps1
(Save -> tv_queue.json)

3) Rip queue
.\run_rip_queue.ps1

## Failure behavior (by design)
- If mapping is uncertain -> job folder moves to:
  C:\MediaPipeline\Ripping_Staging\2_Work_Bench\TV Review
- If MakeMKV errors -> job folder moves to:
  C:\MediaPipeline\Ripping_Staging\2_Work_Bench\Unable_to_Read

## Requirements
- Python 3.x
- MakeMKV CLI installed
- FFmpeg/ffprobe installed and available:
  winget install -e --id Gyan.FFmpeg