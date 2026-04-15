from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from traffic_analytics.config import CountLineConfig, ZoneConfig
from traffic_analytics.geometry import crosses_line, point_in_polygon
from traffic_analytics.tracker_backend import TrackedObject


@dataclass(frozen=True)
class EventRecord:
    event_type: str
    track_id: int
    frame_idx: int
    timestamp_sec: float
    target_name: str = ""
    source_zone: str = ""
    target_zone: str = ""
    movement_label: str = ""
    suppressed_duplicate: bool = False

    def to_csv_row(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 6),
            "target_name": self.target_name,
            "source_zone": self.source_zone,
            "target_zone": self.target_zone,
            "movement_label": self.movement_label,
            "suppressed_duplicate": self.suppressed_duplicate,
        }


@dataclass
class TrackState:
    track_id: int
    class_name: str
    first_frame_idx: int
    last_frame_idx: int
    first_point: tuple[float, float]
    last_point: tuple[float, float]
    frame_count: int = 1
    current_zone: str | None = None
    zone_sequence: list[str] = field(default_factory=list)
    crossed_lines: set[str] = field(default_factory=set)
    entered_zones: set[str] = field(default_factory=set)
    exited_zones: set[str] = field(default_factory=set)
    movement_logged: bool = False
    movement_label: str = ""
    movement_source_zone: str = ""
    movement_target_zone: str = ""

    @property
    def entry_zone(self) -> str | None:
        return self.zone_sequence[0] if self.zone_sequence else None


class AnalyticsEngine:
    def __init__(
        self,
        count_lines: tuple[CountLineConfig, ...],
        zones: tuple[ZoneConfig, ...],
        movement_map: dict[str, dict[str, str]],
    ) -> None:
        self.count_lines = count_lines
        self.zones = zones
        self.movement_map = movement_map

        self.track_states: dict[int, TrackState] = {}
        self.events: list[EventRecord] = []

        self.line_counts: Counter[str] = Counter()
        self.zone_entry_counts: Counter[str] = Counter()
        self.zone_exit_counts: Counter[str] = Counter()
        self.movement_counts: Counter[str] = Counter()
        self.transition_counts: Counter[str] = Counter()
        self.transition_matrix: dict[str, Counter[str]] = {}
        self.duplicate_suppressed_events = 0
        self._finalized = False
        self._unknown_track_ids: list[int] = []

    def process_tracks(
        self,
        tracks: list[TrackedObject],
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        for track in tracks:
            state = self.track_states.get(track.track_id)
            if state is None:
                current_zone = self._resolve_zone(track.point)
                state = TrackState(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    first_frame_idx=frame_idx,
                    last_frame_idx=frame_idx,
                    first_point=track.point,
                    last_point=track.point,
                    current_zone=current_zone,
                )
                self.track_states[track.track_id] = state
                if current_zone is not None:
                    self._record_zone_entry(state, current_zone, frame_idx, timestamp_sec)
                continue

            state.frame_count += 1
            state.last_frame_idx = frame_idx
            self._check_line_crossings(state, track, frame_idx, timestamp_sec)
            current_zone = self._resolve_zone(track.point)
            self._handle_zone_transition(
                state,
                current_zone,
                frame_idx,
                timestamp_sec,
            )
            state.last_point = track.point

    def finalize(self) -> dict[str, object]:
        if self._finalized:
            return self._summary_dict()

        for state in self.track_states.values():
            if not state.movement_logged and (state.zone_sequence or state.crossed_lines):
                self.movement_counts["unknown"] += 1
                self._unknown_track_ids.append(state.track_id)
        self._finalized = True
        return self._summary_dict()

    def get_live_counts(self) -> dict[str, dict[str, int]]:
        return {
            "line_counts": dict(self.line_counts),
            "zone_entry_counts": dict(self.zone_entry_counts),
            "zone_exit_counts": dict(self.zone_exit_counts),
            "movement_counts": dict(self.movement_counts),
        }

    def get_transition_matrix(self) -> dict[str, dict[str, int]]:
        return {
            target_zone: dict(source_counts)
            for target_zone, source_counts in self.transition_matrix.items()
        }

    def get_track_entry_zone(self, track_id: int) -> str | None:
        state = self.track_states.get(track_id)
        if state is None:
            return None
        return state.entry_zone

    def _summary_dict(self) -> dict[str, object]:
        return {
            "line_counts": dict(self.line_counts),
            "zone_entry_counts": dict(self.zone_entry_counts),
            "zone_exit_counts": dict(self.zone_exit_counts),
            "movement_counts": dict(self.movement_counts),
            "transition_counts": dict(self.transition_counts),
            "transition_matrix": self.get_transition_matrix(),
            "duplicate_suppressed_events": self.duplicate_suppressed_events,
            "unknown_track_ids": list(self._unknown_track_ids),
            "analytic_track_count": self._analytic_track_count(),
            "event_count": len(self.events),
        }

    def _analytic_track_count(self) -> int:
        return sum(
            1
            for state in self.track_states.values()
            if state.zone_sequence or state.crossed_lines
        )

    def _check_line_crossings(
        self,
        state: TrackState,
        track: TrackedObject,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        for count_line in self.count_lines:
            if not crosses_line(
                state.last_point,
                track.point,
                count_line.points[0],
                count_line.points[1],
            ):
                continue

            suppressed = count_line.name in state.crossed_lines
            self.events.append(
                EventRecord(
                    event_type="line_crossing",
                    track_id=track.track_id,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    target_name=count_line.name,
                    suppressed_duplicate=suppressed,
                )
            )
            if suppressed:
                self.duplicate_suppressed_events += 1
            else:
                state.crossed_lines.add(count_line.name)
                self.line_counts[count_line.name] += 1

    def _handle_zone_transition(
        self,
        state: TrackState,
        current_zone: str | None,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        previous_zone = state.current_zone
        if current_zone == previous_zone:
            return

        if previous_zone is not None:
            self._record_zone_exit(state, previous_zone, frame_idx, timestamp_sec)
        if current_zone is not None:
            self._record_zone_entry(state, current_zone, frame_idx, timestamp_sec)

        state.current_zone = current_zone

    def _record_zone_entry(
        self,
        state: TrackState,
        zone_name: str,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        suppressed = zone_name in state.entered_zones
        self.events.append(
            EventRecord(
                event_type="zone_entry",
                track_id=state.track_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                target_name=zone_name,
                target_zone=zone_name,
                suppressed_duplicate=suppressed,
            )
        )
        if suppressed:
            self.duplicate_suppressed_events += 1
            return

        state.entered_zones.add(zone_name)
        self.zone_entry_counts[zone_name] += 1

        if not state.zone_sequence or state.zone_sequence[-1] != zone_name:
            state.zone_sequence.append(zone_name)
            self._maybe_record_movement(state, frame_idx, timestamp_sec)

    def _record_zone_exit(
        self,
        state: TrackState,
        zone_name: str,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        suppressed = zone_name in state.exited_zones
        self.events.append(
            EventRecord(
                event_type="zone_exit",
                track_id=state.track_id,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                target_name=zone_name,
                source_zone=zone_name,
                suppressed_duplicate=suppressed,
            )
        )
        if suppressed:
            self.duplicate_suppressed_events += 1
            return

        state.exited_zones.add(zone_name)
        self.zone_exit_counts[zone_name] += 1

    def _maybe_record_movement(
        self,
        state: TrackState,
        frame_idx: int,
        timestamp_sec: float,
    ) -> None:
        if state.movement_logged or len(state.zone_sequence) < 2:
            return

        source_zone = state.zone_sequence[0]
        transition_map = self.movement_map.get(source_zone, {})
        for target_zone in state.zone_sequence[1:]:
            movement_label = transition_map.get(target_zone)
            if movement_label is None:
                continue

            transition_key = f"{source_zone}->{target_zone}"
            state.movement_logged = True
            state.movement_label = movement_label
            state.movement_source_zone = source_zone
            state.movement_target_zone = target_zone

            self.transition_counts[transition_key] += 1
            self.transition_matrix.setdefault(target_zone, Counter())
            self.transition_matrix[target_zone][source_zone] += 1
            self.movement_counts[movement_label] += 1
            self.events.append(
                EventRecord(
                    event_type="movement_classification",
                    track_id=state.track_id,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    target_name=transition_key,
                    source_zone=source_zone,
                    target_zone=target_zone,
                    movement_label=movement_label,
                    suppressed_duplicate=False,
                )
            )
            return

    def _resolve_zone(self, point: tuple[float, float]) -> str | None:
        for zone in self.zones:
            if point_in_polygon(point, zone.polygon):
                return zone.name
        return None
