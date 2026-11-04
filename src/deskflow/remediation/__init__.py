"""Remediation subpackage.

Proposes and applies safe, user-confirmed fixes for diagnosed issues.
All actions require explicit confirmation and support --dry-run.
"""

from deskflow.remediation.actions import (
    ActionResult,
    HeavyTaskGroup,
    clean_app_caches,
    clean_temp,
    close_idle_apps,
    empty_recycle_bin,
    end_busy_process,
    end_heavy_tasks,
    find_heavy_tasks,
    trim_memory,
)
from deskflow.remediation.apps import IdleAppGroup, find_idle_apps
from deskflow.remediation.log import append_fix_log
from deskflow.remediation.recommend import RecommendedAction, recommend_fixes

__all__ = [
    "ActionResult",
    "HeavyTaskGroup",
    "IdleAppGroup",
    "RecommendedAction",
    "append_fix_log",
    "clean_app_caches",
    "clean_temp",
    "close_idle_apps",
    "empty_recycle_bin",
    "end_busy_process",
    "end_heavy_tasks",
    "find_heavy_tasks",
    "find_idle_apps",
    "recommend_fixes",
    "trim_memory",
]
