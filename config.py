from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / 'db' / 'traffic.db'
MODEL_PATH = PROJECT_ROOT / 'runs' / 'detect' / 'people_vehicle_detector-5' / 'weights' / 'best.pt'
DEFAULT_VIDEO_SOURCE = PROJECT_ROOT / 'video.mp4'
ENV_FILE = PROJECT_ROOT / '.env'


def load_dotenv_file(path: Path | str = ENV_FILE) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"\'')

        if key and key not in os.environ:
            os.environ[key] = value


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
