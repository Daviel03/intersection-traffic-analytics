# Intersection Traffic Analytics MVP

This repo is a small, class-project-friendly traffic analytics pipeline for a single intersection scene.

It uses one shared detector (`yolov8n.pt`) and switches only the tracker backend (`ByteTrack` vs `BoT-SORT`) so the comparison stays fair:

`detect -> track -> analyze -> export -> visualize`

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

## Repo Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── trackers/
│   │   ├── bytetrack.yaml
│   │   └── botsort.yaml
│   └── scenes/
│       └── intersection_demo.yaml
├── data/
├── outputs/
├── scripts/
│   ├── run_pipeline.py
│   └── compare_trackers.py
└── src/traffic_analytics/
    ├── pipeline.py
    ├── tracker_backend.py
    ├── analytics.py
    ├── visualization.py
    ├── evaluation.py
    ├── config.py
    ├── geometry.py
    └── io_utils.py
```

## Setup

Python `3.10` to `3.12` is the safest target for Ultralytics and OpenCV.

```bash
pip install -r requirements.txt
```

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

Run optional GT-backed subset evaluation with a local TrackEval checkout:

```bash
python scripts/bootstrap_gt_subset.py --scene intersection_demo --subset long_chunk_subset --frame-start 1660 --frame-end 1839 --classes car
python scripts/run_gt_eval.py --scene intersection_demo --subset short_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_demo --subset interaction_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_demo --subset interaction_wide_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_eval.py --scene intersection_behnam --subset roundabout_turn_subset --trackeval-root /path/to/TrackEval
python scripts/run_gt_suite.py --trackeval-root /path/to/TrackEval
```

The bootstrap helper creates review-ready long subsets from ByteTrack/BoT-SORT overlap so you can scale beyond tiny clips without hand-entering hundreds of boxes. Treat those bootstrap-generated subsets as draft GT that should still be spot-checked before final paper claims.

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
- GT-backed subset outputs: labeled-subset tracking metrics from TrackEval

These layers should not be interpreted the same way. The full-scene metrics tell you how tracker choice changes downstream traffic analytics outcomes. The GT subset provides a limited-scope MOT-style sanity check with standard tracking metrics.

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
    plots/
      movement_counts_<scene>.png
      movement_counts_<scene>.pdf
      total_counts_<scene>.png
      total_counts_<scene>.pdf
      tracking_proxies_<scene>.png
      tracking_proxies_<scene>.pdf
      transitions_<scene>.png
      transitions_<scene>.pdf
    gt/
      gt_eval_summary.csv
      gt_eval_table.tex
      filter_checks.csv
    plots/
      gt_metrics_<scene>_<subset>.png
      gt_metrics_<scene>_<subset>.pdf
```

`metrics_summary.csv` stores one row per `(scene, tracker)` run with:

- total line crossings
- total zone transitions
- left / straight / right / unknown counts
- unknown movement ratio
- average track length
- short-track ratio
- suspected handoff count
- duplicate-suppressed events

`transition_counts.csv` stores one row per named transition such as `far_entry->foreground_exit`.

`run_experiments.py` reuses existing scene outputs by default and reruns only missing scenes unless `--force-rerun` is passed.

`make_plots.py` reads the aggregated CSVs and regenerates the PNG/PDF plots independently, so you do not need to rerun the trackers just to refresh figures.

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

`gt_eval_summary.csv` stores:

- `HOTA`
- `IDF1`
- `MOTA`
- `IDSW`
- `FP`
- `FN`

The GT plot uses `HOTA`, `IDF1`, and `MOTA`. `IDSW`, `FP`, and `FN` remain table-only in v1. When multiple labeled subsets exist for one scene, `make_plots.py` writes one GT figure per `(scene, subset)`.

For Overleaf, the generated `.tex` tables use `booktabs`, so add:

```latex
\usepackage{booktabs}
```

## Notes

- BoT-SORT starts with ReID disabled to keep the baseline lightweight and fair.
- `configs/trackers/bytetrack.yaml` and `configs/trackers/botsort.yaml` follow the official Ultralytics tracker config pattern.
- Full-scene analytics are application-level outcomes, not stand-alone proof of universal tracker quality.
- GT subset results are standard-style tracking evidence, but still limited-scope rather than a full benchmark study.
