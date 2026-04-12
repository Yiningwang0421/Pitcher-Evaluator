from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

FAMILY_ORDER: tuple[str, str, str] = ("fastball", "breaking", "offspeed")

DEFAULT_PITCH_FAMILY_MAPPING: dict[str, str] = {
    "FF": "fastball",
    "FT": "fastball",
    "SI": "fastball",
    "FC": "fastball",
    "SL": "breaking",
    "CU": "breaking",
    "KC": "breaking",
    "SV": "breaking",
    "CH": "offspeed",
    "FS": "offspeed",
    "FO": "offspeed",
    "SC": "offspeed",
}


@dataclass
class FamilyMappingStats:
    total_rows: int
    mapped_rows: int
    dropped_rows: int
    fallback_rows: int


def normalize_pitch_type(value: object) -> str:
    if pd.isna(value):
        return ""
    pitch_type = str(value).strip().upper()
    # Treat Statcast FA as FF to avoid splitting four-seam aliases.
    if pitch_type == "FA":
        return "FF"
    return pitch_type


def map_pitch_type_to_family(
    pitch_type: object,
    mapping: Mapping[str, str],
    drop_unknown_pitch_types: bool,
    unknown_pitch_family_fallback: str,
) -> str | None:
    normalized = normalize_pitch_type(pitch_type)
    if not normalized:
        return None if drop_unknown_pitch_types else unknown_pitch_family_fallback

    family = mapping.get(normalized)
    if family is not None:
        return family

    return None if drop_unknown_pitch_types else unknown_pitch_family_fallback


def add_pitch_family_column(
    df: pd.DataFrame,
    pitch_type_col: str,
    out_col: str,
    mapping: Mapping[str, str],
    drop_unknown_pitch_types: bool,
    unknown_pitch_family_fallback: str,
) -> tuple[pd.DataFrame, FamilyMappingStats]:
    rows = []
    mapped_rows = 0
    dropped_rows = 0
    fallback_rows = 0

    for _, row in df.iterrows():
        family = map_pitch_type_to_family(
            pitch_type=row.get(pitch_type_col),
            mapping=mapping,
            drop_unknown_pitch_types=drop_unknown_pitch_types,
            unknown_pitch_family_fallback=unknown_pitch_family_fallback,
        )
        if family is None:
            dropped_rows += 1
            continue

        normalized_pitch_type = normalize_pitch_type(row.get(pitch_type_col))
        if normalized_pitch_type not in mapping:
            fallback_rows += 1

        copied = row.copy()
        copied[out_col] = family
        rows.append(copied)
        mapped_rows += 1

    out = pd.DataFrame(rows)
    stats = FamilyMappingStats(
        total_rows=int(len(df)),
        mapped_rows=int(mapped_rows),
        dropped_rows=int(dropped_rows),
        fallback_rows=int(fallback_rows),
    )
    return out, stats
