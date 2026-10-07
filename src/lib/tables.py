"""
tables.py
=========
Writing of the csv and markdown result files.
"""
from __future__ import annotations

import csv
import os

import numpy as np


def write_csv(path: str, fields, rows) -> None:
    """Write `rows` (dicts keyed by `fields`) to a csv file, creating its directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_lines(path: str, lines) -> None:
    """Write a list of text lines to a file, creating its directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def fmt(value, digits: int = 3) -> str:
    """Fixed-point text of a number; a hyphen for a missing value (None or NaN)."""
    if value is None or np.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def md_table(header, rows) -> list[str]:
    """Lines of a markdown table; the first column is left-aligned, the others right-aligned."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines
