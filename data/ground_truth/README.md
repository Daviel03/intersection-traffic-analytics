Place optional labeled GT subsets here.

Expected layout:

```text
data/
  ground_truth/
    <scene_name>/
      <subset_name>/
        subset.yaml
        gt_tracks.csv
```

`subset.yaml` fields:

- `scene_name`
- `video_path`
- `frame_start`
- `frame_end`
- `classes`
- optional `description`

`gt_tracks.csv` columns:

- `frame_idx`
- `track_id`
- `class_name`
- `x1`
- `y1`
- `x2`
- `y2`

The GT evaluation script filters both GT and predictions to the same frame range and class list before exporting a TrackEval-compatible MOT-style subset.

Suggested starter subsets in this repo:

- `intersection_demo/short_subset`: easy single-bus sanity-check segment
- `intersection_demo/interaction_subset`: harder multi-car interaction segment
- `intersection_demo/interaction_wide_subset`: wider multi-car interaction window for stronger GT metrics on the street-view clip
- `intersection_demo/long_chunk_subset`: 180-frame consensus-seeded continuous street-view chunk for stronger HOTA/IDF1/MOTA evidence
- `intersection_behnam/roundabout_turn_subset`: overhead roundabout clip with a right-turn path and nearby vehicles

For longer chunks, use the bootstrap helper to seed a review-ready subset from tracker overlap:

```bash
python scripts/bootstrap_gt_subset.py --scene intersection_demo --subset long_chunk_subset --frame-start 1660 --frame-end 1839 --classes car
```

Bootstrap-generated subsets are useful for scaling beyond tiny clips, but they should still be manually reviewed before being described as final GT in a paper.
