import csv
import io
import json
from datetime import datetime
from typing import Any


def isoformat(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return value


def model_to_dict(model: Any, fields: list[str]) -> dict[str, Any]:
    return {field: isoformat(getattr(model, field)) for field in fields}


def rows_to_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, default=str, sort_keys=True) for row in rows) + ("\n" if rows else "")


def rows_to_csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()
