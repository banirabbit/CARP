"""Small, explicit readers for CARP's canonical JSONL execution format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{line_number} is not valid JSON') from error
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def require_execution_fields(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> None:
    for index, record in enumerate(records):
        if not isinstance(record.get('question'), str):
            raise ValueError(f'record {index} has no string question')
        observed = record.get('methods')
        if not isinstance(observed, dict):
            raise ValueError(f'record {index} has no methods object')
        missing = [method for method in methods if method not in observed]
        if missing:
            raise ValueError(f'record {index} is missing methods: {missing}')
        for method in methods:
            values = observed[method]
            if not isinstance(values, dict) or 'f1' not in values or 'total_tokens' not in values:
                raise ValueError(f'record {index}, method {method} needs f1 and total_tokens')


def uplift_targets(records: Sequence[dict[str, Any]], anchor: str, methods: Sequence[str]) -> np.ndarray:
    require_execution_fields(records, methods)
    non_anchor = [method for method in methods if method != anchor]
    return np.asarray(
        [[float(record['methods'][method]['f1']) - float(record['methods'][anchor]['f1']) for method in non_anchor]
         for record in records],
        dtype=np.float32,
    )


def total_token_matrix(records: Sequence[dict[str, Any]], methods: Sequence[str]) -> np.ndarray:
    require_execution_fields(records, methods)
    return np.asarray(
        [[float(record['methods'][method]['total_tokens']) for method in methods] for record in records],
        dtype=np.float64,
    )
