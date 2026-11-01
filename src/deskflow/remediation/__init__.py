"""Remediation subpackage.

Proposes and applies safe, user-confirmed fixes for diagnosed issues.
All actions require explicit confirmation and support --dry-run.
"""

from deskflow.remediation.actions import (
    ActionResult,
    clean_temp,
    close_idle_apps,
    end_busy_process,
    trim_memory,
)
from deskflow.remediation.log import append_fix_log
from deskflow.remediation.recommend import RecommendedAction, recommend_fixes

__all__ = [
    "ActionResult",
    "RecommendedAction",
    "append_fix_log",
    "clean_temp",
    "close_idle_apps",
    "end_busy_process",
    "recommend_fixes",
    "trim_memory",
]
