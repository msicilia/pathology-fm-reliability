"""
names.py
========
Display names of the tasks, used in result tables and figures.
"""
from __future__ import annotations

TASK_NAMES = {
    "nct": "NCT-CRC-HE",
    "lung": "LC25000 lung",
    "lc_colon": "LC25000 colon",
    "breakhis": "BreaKHis",
    "pcam": "PatchCamelyon",
    "sicap": "SICAPv2",
    "panda": "PANDA",
}


def task_name(task: str) -> str:
    return TASK_NAMES[task]
