from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2


_LABEL_MAP = {
    'pedestrian': 'person',
    'person': 'person',
    'people': 'person',
    'car': 'vehicle',
    'cars': 'vehicle',
    'vehicle': 'vehicle',
    'vehicles': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'van': 'vehicle',
}


def normalize_label(label: str) -> str | None:
    return _LABEL_MAP.get(label.strip().lower())


@dataclass(frozen=True)
class TrackedObservation:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float = 0.0


@dataclass
class _TrackState:
    label: str
    track_history: list[tuple[float, float]] = field(default_factory=list)
    counted: bool = False
    missed_frames: int = 0
    seen_frames: int = 0


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _model_class_name(model: object, class_id: int) -> str:
    names = getattr(model, 'names', {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except Exception:
        return str(class_id)


def extract_tracked_observations(model: object, result: object) -> list[TrackedObservation]:
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return []

    track_ids = getattr(boxes, 'id', None)
    xyxy = getattr(boxes, 'xyxy', None)
    classes = getattr(boxes, 'cls', None)
    confs = getattr(boxes, 'conf', None)
    if track_ids is None or xyxy is None or classes is None:
        return []

    track_id_list = track_ids.tolist()
    xyxy_list = xyxy.tolist()
    class_list = classes.tolist()
    conf_list = confs.tolist() if confs is not None else [0.0] * len(xyxy_list)

    observations: list[TrackedObservation] = []
    for track_id, box, class_id, confidence in zip(track_id_list, xyxy_list, class_list, conf_list):
        label = normalize_label(_model_class_name(model, int(class_id)))
        if label is None:
            continue
        x1, y1, x2, y2 = box
        bbox = (float(x1), float(y1), float(x2), float(y2))
        observations.append(
            TrackedObservation(
                track_id=int(track_id),
                label=label,
                bbox=bbox,
                center=_center(bbox),
                confidence=float(confidence),
            )
        )
    return observations


def draw_tracking_overlay(
    frame: object,
    observations: Iterable[TrackedObservation],
    totals: dict[str, int],
    crossed_now: dict[str, int],
    counter: object,
) -> object:
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # Counting lines are intentionally not drawn (kept invisible). The crossing
    # logic in LineCrossCounter still uses them; this only affects the overlay.

    panel_w = 250
    panel_h = 90
    cv2.rectangle(annotated, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(annotated, 0.95, annotated, 0.0, 0, annotated)

    lines = [
        f"Person: {totals['person']}",
        f"Vehicle: {totals['vehicle']}",
    ]
    y = 35
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 30

    for obs in observations:
        x1, y1, x2, y2 = map(int, obs.bbox)
        
        if obs.label == 'person':
            color = (0, 255, 0)
            label = 'person'
        else:
            color = (0, 165, 255)
            label = 'vehicle'
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


class LineCrossCounter:
    def __init__(
        self,
        *,
        pedestrian_left_x: float = 0.01,
        pedestrian_right_x: float = 0.99,
        pedestrian_bottom_y: float = 0.95,
        vehicle_line_x: float = 0.95,
        max_missed_frames: int = 30,
        min_hits: int = 3,
        debug: bool = True,
    ) -> None:
        self.pedestrian_left_x = pedestrian_left_x
        self.pedestrian_right_x = pedestrian_right_x
        self.pedestrian_bottom_y = pedestrian_bottom_y
        self.vehicle_line_x = vehicle_line_x
        self.max_missed_frames = max_missed_frames
        # A track must be seen at least this many frames before it can be counted.
        # Real objects persist; flickering false positives (e.g. a wall briefly
        # misdetected) disappear before reaching the threshold and are ignored.
        self.min_hits = min_hits
        self.debug = debug
        self._tracks: dict[int, _TrackState] = {}
        self._totals = {'person': 0, 'vehicle': 0}
        self._pending = {'person': 0, 'vehicle': 0}
        self._debug_log: list[str] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._totals = {'person': 0, 'vehicle': 0}
        self._pending = {'person': 0, 'vehicle': 0}

    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def drain_pending(self) -> dict[str, int]:
        pending = dict(self._pending)
        self._pending = {'person': 0, 'vehicle': 0}
        return pending

    def get_debug_log(self) -> list[str]:
        log = list(self._debug_log)
        self._debug_log.clear()
        return log

    def line_positions(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        pedestrian_left_x_px = int(frame_width * self.pedestrian_left_x)
        pedestrian_right_x_px = int(frame_width * self.pedestrian_right_x)
        pedestrian_bottom_y_px = int(frame_height * self.pedestrian_bottom_y)
        vehicle_line_x_px = int(frame_width * self.vehicle_line_x)
        return pedestrian_left_x_px, pedestrian_right_x_px, pedestrian_bottom_y_px, vehicle_line_x_px

    def update(
        self,
        observations: Iterable[TrackedObservation],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, int]:
        pedestrian_left_x_px, pedestrian_right_x_px, pedestrian_bottom_y_px, vehicle_line_x_px = self.line_positions(frame_width, frame_height)

        current_ids: set[int] = set()
        crossed_now = {'person': 0, 'vehicle': 0}

        for obs in observations:
            current_ids.add(obs.track_id)
            cx, cy = obs.center

            state = self._tracks.get(obs.track_id)
            if state is None:
                state = _TrackState(
                    label=obs.label,
                    track_history=[(cx, cy)],
                    counted=False,
                    missed_frames=0,
                    seen_frames=1,
                )
                self._tracks[obs.track_id] = state
                continue

            state.track_history.append((cx, cy))
            if len(state.track_history) > 50:
                state.track_history.pop(0)

            state.label = obs.label
            state.missed_frames = 0
            state.seen_frames += 1

            if not state.counted and state.seen_frames >= self.min_hits and len(state.track_history) >= 2:
                prev_cx, prev_cy = state.track_history[-2]

                if state.label == 'person':
                    crossed = False

                    if prev_cx < pedestrian_left_x_px <= cx:
                        crossed = True
                    elif prev_cx > pedestrian_left_x_px >= cx:
                        crossed = True
                    elif prev_cx < pedestrian_right_x_px <= cx:
                        crossed = True
                    elif prev_cx > pedestrian_right_x_px >= cx:
                        crossed = True
                    elif prev_cy < pedestrian_bottom_y_px <= cy:
                        crossed = True
                    elif prev_cy > pedestrian_bottom_y_px >= cy:
                        crossed = True

                    if crossed:
                        state.counted = True
                        crossed_now['person'] += 1
                        self._totals['person'] += 1
                        self._pending['person'] += 1

                elif state.label == 'vehicle':
                    if prev_cx < vehicle_line_x_px <= cx:
                        state.counted = True
                        crossed_now['vehicle'] += 1
                        self._totals['vehicle'] += 1
                        self._pending['vehicle'] += 1
                    elif prev_cx > vehicle_line_x_px >= cx:
                        state.counted = True
                        crossed_now['vehicle'] += 1
                        self._totals['vehicle'] += 1
                        self._pending['vehicle'] += 1

        expired_ids: list[int] = []
        for track_id, state in self._tracks.items():
            if track_id in current_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames > self.max_missed_frames:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self._tracks.pop(track_id, None)

        return crossed_now
