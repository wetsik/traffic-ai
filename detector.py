from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import DEFAULT_VIDEO_SOURCE, MODEL_PATH, load_dotenv_file
from db.database import insert_detection
from tracking import LineCrossCounter, extract_tracked_observations


CONFIDENCE_THRESHOLD = 0.7
WRITE_INTERVAL_SECONDS = 10.0


def _resolve_source(source: str) -> str | int:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def _is_file_source(source: str | int) -> bool:
    return isinstance(source, str) and Path(source).exists()


def run_detector(
    source: str | None = None,
    model_path: str | Path | None = None,
    confidence: float = CONFIDENCE_THRESHOLD,
) -> None:
    load_dotenv_file()

    resolved_source = source or os.getenv('VIDEO_SOURCE', str(DEFAULT_VIDEO_SOURCE.name))
    resolved_model_path = Path(model_path or MODEL_PATH)

    if not resolved_model_path.exists():
        raise FileNotFoundError(f'Model not found: {resolved_model_path}')

    model = YOLO(str(resolved_model_path))
    capture = cv2.VideoCapture(_resolve_source(resolved_source))
    if not capture.isOpened():
        raise RuntimeError(f'Unable to open video source: {resolved_source}')

    pass_counter = LineCrossCounter()
    next_write_at = time.monotonic() + WRITE_INTERVAL_SECONDS

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if _is_file_source(_resolve_source(resolved_source)):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            results = model.track(
                frame,
                conf=confidence,
                persist=True,
                tracker='bytetrack.yaml',
                verbose=False,
                imgsz=640,
                workers=0,
            )
            observations = extract_tracked_observations(model, results[0])
            pass_counter.update(observations, frame.shape[1], frame.shape[0])

            now = time.monotonic()
            if now >= next_write_at:
                counts = pass_counter.drain_pending()
                if any(counts.values()):
                    insert_detection(
                        pedestrians=int(counts['pedestrians']),
                        cars=int(counts['cars']),
                        vehicles=int(counts['vehicles']),
                    )
                totals = pass_counter.totals()
                print(
                    f'[writer] pedestrians={counts["pedestrians"]} '
                    f'cars={counts["cars"]} vehicles={counts["vehicles"]} '
                    f'total_pedestrians={totals["pedestrians"]} '
                    f'total_cars={totals["cars"]} '
                    f'total_vehicles={totals["vehicles"]}'
                )
                next_write_at = now + WRITE_INTERVAL_SECONDS
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detector()
