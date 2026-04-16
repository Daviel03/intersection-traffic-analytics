# LiDAR Evidence Format

This folder is for lightweight, precomputed LiDAR evidence used by the late-fusion pipeline.

In practice, the easiest workflow is:

1. keep the large raw camera / LiDAR files in Google Drive
2. use `scripts/link_external_data.py` from Colab to link them into this repo layout
3. run the existing camera-only and fusion scripts against the linked paths

Expected per-scene path:

```text
data/
  lidar/
    <scene_name>/
      evidence.csv
      raw/
```

Required CSV columns:

- `frame_idx`
- `object_id`
- `support_score`

Recommended columns:

- `x1`
- `y1`
- `x2`
- `y2`
- `center_x`
- `center_y`
- `range_m`

Notes:

- If bbox columns are present, fusion can use IoU and point proximity.
- If only centers are present, fusion falls back to point-distance matching.
- If no real LiDAR evidence file exists, `scripts/run_fusion_experiments.py` can generate a mock motion-based evidence file from camera tracks for a lightweight class-project demo.
- If you only have raw LiDAR data, keep it under `data/lidar/<scene_name>/raw/` and preprocess it into `evidence.csv` before running the current fusion MVP.
