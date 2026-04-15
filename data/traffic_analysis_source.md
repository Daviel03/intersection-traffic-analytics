# traffic_analysis.mov source

- Title used by reference repo: `traffic_analysis.mov`
- Reference repo: `https://github.com/Behnam-Asadi/YOLOv8-traffic-analysis`
- Download script reference: `https://github.com/Behnam-Asadi/YOLOv8-traffic-analysis/blob/main/setup.sh`
- Download URL used by the reference repo: `https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W`
- Downloaded into this workspace on: `2026-04-14`
- Local file: `data/traffic_analysis.mov`

## Geometry provenance

The current `configs/scenes/intersection_behnam.yaml` started from the Behnam repo's rectangular entry and exit concept, which comes from:

- `https://github.com/Behnam-Asadi/YOLOv8-traffic-analysis/blob/main/video_processing/utils.py`

Originally:

- `ZONE_IN_POLYGONS` defined four directional entry rectangles
- `ZONE_OUT_POLYGONS` defined four directional exit rectangles

## Adaptation notes

- The reference repo uses zone triggering with the bounding-box center.
- Our pipeline uses the bounding-box bottom-center for all line and zone logic.
- The current YAML is no longer a direct rectangle import. It was retuned around the visible bottom-center traffic footprint so the scene functions as a better second comparison scene in this repo.
- The current count lines and zones are therefore adapted traffic corridors, not an exact copy of the original `ZONE_IN_POLYGONS` and `ZONE_OUT_POLYGONS`.
- The current `movement_map` keeps only the movement semantics that the visible camera view supports reliably under bottom-center triggering.

## Repo note

The project currently ignores local video files in `data/`, so this clip exists in the workspace but will not be committed unless you change `.gitignore` or use a large-file workflow.
