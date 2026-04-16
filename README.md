# Intersection Traffic Analytics MVP

This repo is a small, class-project-friendly traffic analytics pipeline for intersection scenes.

It uses one shared detector (`yolov8n.pt`) and switches only the tracker backend (`ByteTrack` vs `BoT-SORT`) so the comparison stays fair:

`detect -> track -> analyze -> export -> visualize`

On top of that camera-only baseline, the repo now also includes a lightweight camera+LiDAR late-fusion layer that can reuse the same downstream analytics and experiment tables without turning the project into a heavy multimodal stack.

## Reference Backbone

This project is intentionally built around two external references instead of reimplementing the whole stack from scratch.

- Ultralytics YOLO: the main backbone for detection, tracking, tracker switching, and tracker YAML patterns
- Behnam-Asadi/YOLOv8-traffic-analysis: the analytics-layer reference for entry/exit zone logic, movement summaries, and traffic-oriented overlays

Useful links:

- `https://github.com/ultralytics/ultralytics`
- `https://github.com/Behnam-Asadi/YOLOv8-traffic-analysis`

## What Phase 1 Does

- runs one local traffic video clip
- tracks vehicles with either ByteTrack or BoT-SORT
- counts line crossings
- logs zone entries and exits
- classifies movements as `left`, `straight`, `right`, or `unknown`
- prevents duplicate counts by remembering which events each track ID already triggered
- colors tracks by their first detected entry zone once that information is known
- renders transition counts near the exit zones, inspired by the Behnam-Asadi traffic-analysis overlay style
- exports an annotated video, per-frame tracks, event logs, and a run summary
- compares both trackers on the same scene config

## Current Status

Phase 1 is working and already includes:

- two full-scene comparison runs: `intersection_demo` and `intersection_behnam`
- scene-calibration helpers for previewing overlays and clicking YAML-ready points
- experiment aggregation into CSV, LaTeX tables, and PNG/PDF plots
- optional camera+LiDAR late-fusion experiments using precomputed or lightweight mock LiDAR evidence
- GT-backed subset evaluation with TrackEval-compatible exports
- multiple included GT subsets, from a tiny sanity check to a longer continuous chunk

## Repo Layout

```text
.
|- README.md
|- requirements.txt
|- .gitignore
|- configs/
|  |- default.yaml
|  |- trackers/
|  |  |- bytetrack.yaml
|  |  `- botsort.yaml
|  `- scenes/
|     |- intersection_demo.yaml
|     `- intersection_behnam.yaml
|- data/
|  |- lidar/
|  |  `- README.md
|  `- ground_truth/
|     |- intersection_demo/
|     `- intersection_behnam/
|- outputs/
|- scripts/
|  |- run_pipeline.py
|  |- compare_trackers.py
|  |- link_external_data.py
|  |- export_scene_preview.py
|  |- pick_scene_points.py
|  |- run_experiments.py
|  |- run_fusion_experiments.py
|  |- make_plots.py
|  |- run_gt_eval.py
|  |- run_gt_suite.py
|  `- bootstrap_gt_subset.py
|- notebooks/
|  `- colab_runner.ipynb
|- tests/
`- src/traffic_analytics/
   |- pipeline.py
   |- tracker_backend.py
   |- analytics.py
   |- visualization.py
   |- evaluation.py
   |- experiments.py
   |- external_data.py
   |- lidar.py
   |- fusion.py
   |- fusion_experiments.py
   |- plotting.py
   |- gt_eval.py
   |- gt_bootstrap.py
   |- config.py
   |- geometry.py
   `- io_utils.py
```

## Setup

Python `3.10` to `3.12` is the safest target for Ultralytics and OpenCV.

The commands below use `python`, but on Windows `py -3.11` is a good default if you have multiple Python versions installed.

```bash
pip install -r requirements.txt
```

`yolov8n.pt`, raw videos, generated outputs, and local TrackEval clones are intentionally not tracked in Git. If the YOLO weights are not already present, Ultralytics can download them on first run.

Place one short local intersection clip at:

```text
data/intersection_demo.mp4
```

Then edit the scene geometry in `configs/scenes/intersection_demo.yaml` so the lines and polygons match your camera view.

A second reference scene adapted from `Behnam-Asadi/YOLOv8-traffic-analysis` is also supported with the local clip:

```text
data/traffic_analysis.mov
```

Its imported rectangular zone geometry lives in `configs/scenes/intersection_behnam.yaml`.

## Quick Start

If you want the shortest useful path after cloning:

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run both trackers on both scenes

```bash
python scripts/compare_trackers.py --scene configs/scenes/intersection_demo.yaml
python scripts/compare_trackers.py --scene configs/scenes/intersection_behnam.yaml
```

3. Aggregate experiment tables and plots

```bash
python scripts/run_experiments.py
python scripts/make_plots.py
```

Optional: add the late-fusion layer and regenerate the experiment outputs with a fused system variant:

```bash
python scripts/run_fusion_experiments.py --make-plots
```

4. Run the GT-backed evaluation layer if TrackEval is available locally

```bash
python scripts/run_gt_suite.py --trackeval-root third_party/TrackEval
```

That sequence gives you:

- per-scene tracker outputs under `outputs/<scene>/`
- aggregate CSV and LaTeX summaries under `outputs/experiments/`
- full-scene plots and GT plots under `outputs/experiments/plots/`

## Colab / Drive Workflow

If your camera or LiDAR files are too large to work with comfortably on your laptop, the cleanest setup is:

- GitHub repo = code, configs, tests, small GT subsets
- Google Drive = raw videos, raw LiDAR, large evidence files, large outputs
- Colab = runner with optional GPU

The repo includes:

- [colab_runner.ipynb](/e:/No Hands No Problem- Autonomous Driving/notebooks/colab_runner.ipynb)
- [link_external_data.py](/e:/No Hands No Problem- Autonomous Driving/scripts/link_external_data.py)

The helper script links or copies external data into the repo layout so the existing scene YAMLs work without hardcoding your Drive paths into the repo.

Typical Colab flow:

```bash
python scripts/link_external_data.py --scene intersection_demo --camera-source /content/drive/MyDrive/traffic_data/intersection_demo.mp4 --lidar-source /content/drive/MyDrive/traffic_data/intersection_demo_evidence.csv --force
python scripts/compare_trackers.py --scene configs/scenes/intersection_demo.yaml --output-root /content/drive/MyDrive/traffic_analytics_outputs
python scripts/run_fusion_experiments.py --scenes intersection_demo --output-root /content/drive/MyDrive/traffic_analytics_outputs --make-plots
```

If you prefer a scene copy that records the linked paths explicitly, add:

```bash
python scripts/link_external_data.py --scene intersection_demo --camera-source /content/drive/MyDrive/traffic_data/intersection_demo.mp4 --lidar-source /content/drive/MyDrive/traffic_data/intersection_demo_evidence.csv --scene-copy configs/scenes/intersection_demo_colab.yaml --force
```

That writes a scene YAML copy with updated `video_path` and optional `lidar_evidence_path`.

## Usage

Run one tracker:

```bash
python scripts/run_pipeline.py --scene configs/scenes/intersection_demo.yaml --tracker bytetrack
python scripts/run_pipeline.py --scene configs/scenes/intersection_demo.yaml --tracker botsort
```

Run both trackers and generate comparison outputs:

```bash
python scripts/compare_trackers.py --scene configs/scenes/intersection_demo.yaml
```

Export one scene-preview frame with just the geometry overlays:

```bash
python scripts/export_scene_preview.py --scene configs/scenes/intersection_demo.yaml --frame-index 960
```

Open one frame and click points to build YAML-ready coordinates:

```bash
python scripts/pick_scene_points.py --scene configs/scenes/intersection_demo.yaml --frame-index 960 --kind active-area
python scripts/pick_scene_points.py --scene configs/scenes/intersection_demo.yaml --frame-index 960 --kind zone --name right_exit
python scripts/pick_scene_points.py --scene configs/scenes/intersection_demo.yaml --frame-index 960 --kind line --name right_exit_line
```

Link large external camera / LiDAR data into the repo layout, which is especially handy in Colab:

```bash
python scripts/link_external_data.py --scene intersection_demo --camera-source /absolute/path/to/intersection_demo.mp4 --lidar-source /absolute/path/to/evidence.csv --force
```

Run the Behnam-based reference scene:

```bash
python scripts/run_pipeline.py --scene configs/scenes/intersection_behnam.yaml --tracker bytetrack
python scripts/run_pipeline.py --scene configs/scenes/intersection_behnam.yaml --tracker botsort
python scripts/compare_trackers.py --scene configs/scenes/intersection_behnam.yaml
```

Optional demo polish:

```bash
python scripts/run_pipeline.py --scene configs/scenes/intersection_demo.yaml --tracker bytetrack --save-trails
```

Aggregate full-scene experiment CSVs and LaTeX tables:

```bash
python scripts/run_experiments.py
```

Generate paper-ready plots from the aggregated CSVs:

```bash
python scripts/make_plots.py
```

Run the camera+LiDAR late-fusion experiment layer:

```bash
python scripts/run_fusion_experiments.py
python scripts/run_fusion_experiments.py --fusion-trackers bytetrack botsort --make-plots
```

If you already have per-scene LiDAR evidence CSV files under `data/lidar/<scene>/evidence.csv`, the fusion script will use them. If not, it falls back to a lightweight mock motion-based evidence generator unless `--no-mock-lidar` is passed.

Run optional GT-backed subset evaluation with a local TrackEval checkout:

```bash
python scripts/bootstrap_gt_subset.py --scene intersection_demo --subset long_chunk_subset --frame-start 1660 --frame-end 1839 --classes car
python scripts/run_gt_eval.py --scene intersection_demo --subset short_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_demo --subset short_subset --system-variants camera_bytetrack camera_lidar_bytetrack_fusion --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_demo --subset interaction_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_demo --subset interaction_wide_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_behnam --subset roundabout_turn_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_suite.py --trackeval-root /path/to/TrackEval
```

The bootstrap helper creates review-ready long subsets from ByteTrack/BoT-SORT overlap so you can scale beyond tiny clips without hand-entering hundreds of boxes. Treat those bootstrap-generated subsets as draft GT that should still be spot-checked before final paper claims.

Included GT subsets in this repo:

- `intersection_demo/short_subset`: easy single-bus sanity check
- `intersection_demo/interaction_subset`: compact multi-car interaction
- `intersection_demo/interaction_wide_subset`: wider local interaction window
- `intersection_demo/long_chunk_subset`: 180-frame continuous chunk generated from tracker consensus
- `intersection_behnam/roundabout_turn_subset`: overhead turn-focused subset

## Scene YAML

Phase 1 required fields:

- `video_path`
- `output_name`
- `target_classes`
- `count_lines`
- `zones`
- `movement_map`

Optional Phase 1 fields:

- `active_area`: ignore detections outside the usable intersection region
- `model`: override the default detector from `configs/default.yaml`
- `analytics_classes`: optional subset of `target_classes` that contribute to line counts, zone events, and movement summaries
- `lidar_evidence_path`: optional path to a precomputed LiDAR evidence CSV for the fusion runner

`target_classes` must be class names, not COCO indices. Example:

```yaml
target_classes:
  - person
  - bicycle
  - car
  - bus
  - truck
  - motorcycle
```

If you want to track more road users without letting them affect the vehicle-flow summaries, set `analytics_classes` separately:

```yaml
analytics_classes:
  - bicycle
  - car
  - bus
  - truck
  - motorcycle
```

`movement_map` is a named transition map from entry zone to exit zone. Example:

```yaml
movement_map:
  north_entry:
    west_exit: left
    south_exit: straight
    east_exit: right
```

In this example, a track that first enters `north_entry` and later enters `west_exit` is labeled `left`.

Tracks that cross a count line but never form a valid entry -> exit pair still keep their line counts. Their movement label stays `unknown`.

`configs/scenes/intersection_behnam.yaml` is an imported reference scene: its rectangles come from the Behnam-Asadi repo, while the line definitions and movement labels are adapted to fit this project's shared line-counting and turn-classification pipeline.

## Geometry Rule

All line crossing and zone logic use the **bottom-center** of the tracked bounding box, not the raw bbox center. For vehicles, that point is usually more stable for road-contact reasoning.

## Duplicate Suppression

A duplicate-suppressed event is an event that would have been counted again for the same track and the same target line or zone, but was ignored because that track had already triggered that event.

## Outputs

Each tracker writes to its own folder:

```text
outputs/
  intersection_demo/
    bytetrack/
      annotated.mp4
      tracks.csv
      events.csv
      summary.json
    botsort/
      annotated.mp4
      tracks.csv
      events.csv
      summary.json
    comparison/
      comparison.csv
      comparison.json
      quick_comparison.json
      comparison.md
```

`events.csv` uses these columns:

- `event_type`
- `track_id`
- `frame_idx`
- `timestamp_sec`
- `target_name`
- `source_zone`
- `target_zone`
- `movement_label`
- `suppressed_duplicate`

Typical `event_type` values are:

- `line_crossing`
- `zone_entry`
- `zone_exit`
- `movement_classification`

## Comparison Metrics

The tracker comparison is split into two layers.

Continuity proxies:

- unique track count
- average track length
- short-track ratio
- suspected ID handoff count
- duplicate-suppressed events

Downstream analytics:

- total line count differences
- left / straight / right / unknown count differences
- per-transition count differences such as `far_entry->left_exit`

The run summaries also include a `transition_matrix` nested structure for presentation-friendly exit-zone summaries.
The quick comparison exports also include per-class track counts and frame-level detection counts so you can immediately see how much of the scene is vehicles versus pedestrians or cyclists.

## Evaluation Layers

The report-facing evaluation stack is intentionally split into two layers:

- full-scene experiment outputs: application-level traffic analytics metrics from the complete intersection clips
- optional camera+LiDAR late-fusion outputs: the same downstream analytics after conservative late fusion
- GT-backed subset outputs: labeled-subset tracking metrics from TrackEval

These layers should not be interpreted the same way. The full-scene metrics tell you how tracker choice changes downstream traffic analytics outcomes. The GT subset provides a limited-scope MOT-style sanity check with standard tracking metrics.

## Camera + LiDAR Late Fusion

The fusion extension is intentionally lightweight.

The camera-only baselines stay exactly as they are:

- `camera_bytetrack`
- `camera_botsort`

The fused layer adds one or more system variants such as:

- `camera_lidar_bytetrack_fusion`
- `camera_lidar_botsort_fusion`

The v1 fusion flow is:

1. reuse existing camera track outputs
2. load synchronized or precomputed LiDAR evidence
3. match LiDAR evidence to camera tracks using bbox IoU and bottom-center proximity
4. confirm or suppress tracks conservatively
5. rerun the same traffic analytics logic on the fused track set

This keeps the comparison controlled: same scene, same detector, same class filters, same movement logic, same geometry; only the system variant changes.

Expected LiDAR evidence path:

```text
data/
  lidar/
    <scene_name>/
      evidence.csv
```

See [data/lidar/README.md](/e:/No%20Hands%20No%20Problem-%20Autonomous%20Driving/data/lidar/README.md) for the CSV contract.

You can also point a scene YAML directly at an evidence file with:

```yaml
lidar_evidence_path: data/lidar/intersection_demo/evidence.csv
```

If no real LiDAR evidence is available, `run_fusion_experiments.py` can generate a mock motion-based evidence file from the camera tracks. That is useful for exercising the interface and experiment layer, but it should be described honestly in a paper as a lightweight placeholder rather than a real sensor benchmark.

## Experiment Outputs

The experiment scripts write to:

```text
outputs/
  experiments/
    metrics_summary.csv
    transition_counts.csv
    tables/
      analytics_table.tex
      continuity_table.tex
      fusion_table.tex
    plots/
      movement_counts_<scene>.png
      movement_counts_<scene>.pdf
      total_counts_<scene>.png
      total_counts_<scene>.pdf
      tracking_proxies_<scene>.png
      tracking_proxies_<scene>.pdf
      transitions_<scene>.png
      transitions_<scene>.pdf
      fusion_diagnostics_<scene>.png
      fusion_diagnostics_<scene>.pdf
      gt_metrics_<scene>_<subset>.png
      gt_metrics_<scene>_<subset>.pdf
    gt/
      gt_eval_summary.csv
      gt_eval_table.tex
      filter_checks.csv
      workspace/
```

`metrics_summary.csv` stores one row per `(scene, system_variant)` run with:

- `system_variant`
- `tracker_name`
- `fusion_enabled`
- total line crossings
- total zone transitions
- left / straight / right / unknown counts
- unknown movement ratio
- average track length
- short-track ratio
- suspected handoff count
- duplicate-suppressed events
- LiDAR-supported track count
- LiDAR-unsupported track count
- fused confirmation events
- suppressed camera-only tracks
- average LiDAR support ratio

`transition_counts.csv` stores one row per named transition such as `far_entry->foreground_exit`, keyed by `scene_name` and `system_variant`.

`run_experiments.py` reuses existing scene outputs by default and reruns only missing scenes unless `--force-rerun` is passed.

`run_fusion_experiments.py` preserves the camera-only baselines, adds one or more fused system variants, then rewrites the same aggregate CSV / table / plot outputs so the paper-facing comparison can include all variants together.

`make_plots.py` reads the aggregated CSVs and regenerates the PNG/PDF plots independently, so you do not need to rerun the trackers just to refresh figures.

`filter_checks.csv` is the manual sanity-check artifact for GT evaluation. It records the exact frame range, classes, row counts, and track counts used for both GT and predictions before TrackEval runs.

## GT Subset Format

The optional GT-backed evaluation expects:

```text
data/
  ground_truth/
    <scene_name>/
      <subset_name>/
        subset.yaml
        gt_tracks.csv
```

`subset.yaml` required fields:

- `scene_name`
- `video_path`
- `frame_start`
- `frame_end`
- `classes`

Optional metadata:

- `description`

`gt_tracks.csv` columns:

- `frame_idx`
- `track_id`
- `class_name`
- `x1`
- `y1`
- `x2`
- `y2`

The GT script filters both GT and predictions to the same frame range and class list before TrackEval runs. The selected classes are then collapsed into one MOT-style evaluation class after filtering so TrackEval can score association quality on the chosen objects.

By default the GT path evaluates the camera-only system variants derived from the tracker names. You can also point it at fused outputs with `--system-variants`, for example:

```bash
python scripts/run_gt_eval.py --scene intersection_demo --subset short_subset --system-variants camera_bytetrack camera_lidar_bytetrack_fusion --trackeval-root /path/to/TrackEval
```

`gt_eval_summary.csv` stores:

- `system_variant`
- `HOTA`
- `IDF1`
- `MOTA`
- `IDSW`
- `FP`
- `FN`

The GT plot uses `HOTA`, `IDF1`, and `MOTA`. `IDSW`, `FP`, and `FN` remain table-only in v1. When multiple labeled subsets exist for one scene, `make_plots.py` writes one GT figure per `(scene, subset)`.

For longer continuous chunks, the repo also includes a consensus bootstrap path:

- `bootstrap_gt_subset.py` matches ByteTrack and BoT-SORT boxes by IoU
- averaged boxes are written into a draft `gt_tracks.csv`
- the result is useful for scaling beyond tiny subsets, but it should still be reviewed before you describe it as final GT in a paper

This is a pragmatic class-project compromise: full-scene analytics run on the entire clip, while MOT-style GT metrics are computed on smaller labeled or bootstrap-reviewed subsets.

For Overleaf, the generated `.tex` tables use `booktabs`, so add:

```latex
\usepackage{booktabs}
```

## Notes

- BoT-SORT starts with ReID disabled to keep the baseline lightweight and fair.
- `configs/trackers/bytetrack.yaml` and `configs/trackers/botsort.yaml` follow the official Ultralytics tracker config pattern.
- Full-scene analytics are application-level outcomes, not stand-alone proof of universal tracker quality.
- Fusion diagnostics are lightweight support metrics for this class project, not stand-alone evidence of production-grade AV fusion quality.
- GT subset results are standard-style tracking evidence, but still limited-scope rather than a full benchmark study.
