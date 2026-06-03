from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.database import upsert_daily_summary


WEEKDAY_PED_PROFILE = [
    0.020, 0.015, 0.012, 0.012, 0.015, 0.025, 0.050, 0.090, 0.130, 0.110, 0.070, 0.060,
    0.060, 0.065, 0.070, 0.080, 0.095, 0.120, 0.110, 0.075, 0.040, 0.030, 0.018, 0.013,
]
WEEKDAY_CAR_PROFILE = [
    0.020, 0.014, 0.010, 0.010, 0.012, 0.025, 0.060, 0.110, 0.140, 0.120, 0.075, 0.060,
    0.055, 0.055, 0.065, 0.085, 0.115, 0.140, 0.135, 0.075, 0.030, 0.020, 0.016, 0.012,
]
WEEKDAY_VEHICLE_PROFILE = [
    0.018, 0.013, 0.010, 0.010, 0.012, 0.020, 0.045, 0.085, 0.110, 0.105, 0.080, 0.070,
    0.070, 0.075, 0.080, 0.095, 0.115, 0.130, 0.125, 0.095, 0.055, 0.040, 0.025, 0.022,
]

WEEKEND_PED_PROFILE = [
    0.015, 0.012, 0.010, 0.010, 0.012, 0.018, 0.030, 0.045, 0.060, 0.070, 0.080, 0.090,
    0.100, 0.105, 0.105, 0.100, 0.090, 0.080, 0.070, 0.055, 0.040, 0.030, 0.020, 0.018,
]
WEEKEND_CAR_PROFILE = [
    0.018, 0.015, 0.012, 0.012, 0.014, 0.020, 0.035, 0.050, 0.060, 0.070, 0.075, 0.080,
    0.090, 0.095, 0.095, 0.095, 0.085, 0.080, 0.075, 0.060, 0.045, 0.030, 0.023, 0.021,
]
WEEKEND_VEHICLE_PROFILE = [
    0.018, 0.015, 0.012, 0.012, 0.014, 0.018, 0.030, 0.045, 0.055, 0.065, 0.075, 0.085,
    0.095, 0.100, 0.100, 0.095, 0.090, 0.085, 0.080, 0.065, 0.050, 0.035, 0.025, 0.021,
]


def _normalize(values: Iterable[float]) -> list[float]:
    values = list(values)
    total = sum(values)
    return [value / total for value in values]


WEEKDAY_PED_PROFILE = _normalize(WEEKDAY_PED_PROFILE)
WEEKDAY_CAR_PROFILE = _normalize(WEEKDAY_CAR_PROFILE)
WEEKDAY_VEHICLE_PROFILE = _normalize(WEEKDAY_VEHICLE_PROFILE)
WEEKEND_PED_PROFILE = _normalize(WEEKEND_PED_PROFILE)
WEEKEND_CAR_PROFILE = _normalize(WEEKEND_CAR_PROFILE)
WEEKEND_VEHICLE_PROFILE = _normalize(WEEKEND_VEHICLE_PROFILE)


def _simulate_total(target: int, profile: list[float], rng: random.Random) -> int:
    total = 0
    jitter = max(1, int(target * 0.03 / len(profile)))
    for weight in profile:
        sample = target * weight + rng.randint(-jitter, jitter)
        total += max(0, int(round(sample)))
    return total


def _day_targets(day: date) -> tuple[int, int, int]:
    weekend = day.weekday() >= 5
    day_number = day.toordinal()
    trend = 1.0 + ((day_number % 9) - 4) * 0.015
    seasonal = 1.0 + 0.07 * math.sin(day_number / 4.5)

    if weekend:
        ped_base, car_base, vehicle_base = 1700, 5200, 2500
        multiplier = 0.82
    else:
        ped_base, car_base, vehicle_base = 2600, 7800, 3400
        multiplier = 1.0 + (0.03 * (3 - abs(3 - day.weekday())))

    pedestrians = int(ped_base * trend * seasonal * multiplier)
    cars = int(car_base * trend * seasonal * multiplier)
    vehicles = int(vehicle_base * trend * seasonal * multiplier)
    return pedestrians, cars, vehicles


def generate_day(day: date, rng: random.Random) -> tuple[int, int, int]:
    weekend = day.weekday() >= 5
    ped_target, car_target, vehicle_target = _day_targets(day)

    if weekend:
        ped_profile = WEEKEND_PED_PROFILE
        car_profile = WEEKEND_CAR_PROFILE
        vehicle_profile = WEEKEND_VEHICLE_PROFILE
    else:
        ped_profile = WEEKDAY_PED_PROFILE
        car_profile = WEEKDAY_CAR_PROFILE
        vehicle_profile = WEEKDAY_VEHICLE_PROFILE

    pedestrians = _simulate_total(ped_target, ped_profile, rng)
    cars = _simulate_total(car_target, car_profile, rng)
    vehicles = _simulate_total(vehicle_target, vehicle_profile, rng)
    return pedestrians, cars, vehicles


def seed_last_30_days(days: int = 30, seed: int = 42) -> list[dict[str, int | str]]:
    rng = random.Random(seed)
    start_day = date.today() - timedelta(days=days - 1)
    inserted: list[dict[str, int | str]] = []

    for offset in range(days):
        day = start_day + timedelta(days=offset)
        pedestrians, cars, vehicles = generate_day(day, rng)
        upsert_daily_summary(day, pedestrians, cars, vehicles)
        inserted.append(
            {
                'date': day.isoformat(),
                'pedestrians': pedestrians,
                'cars': cars,
                'vehicles': vehicles,
            }
        )

    return inserted


def main() -> None:
    rows = seed_last_30_days()
    print(f'Seeded {len(rows)} days of synthetic traffic data into the daily summary table.')
    print(f"Date range: {rows[0]['date']} -> {rows[-1]['date']}")


if __name__ == '__main__':
    main()
