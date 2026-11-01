"""Performance evaluation metrics, breakdowns, and block bootstrap calculations."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from deskflow.features.registry import SLOW_THRESHOLD


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slow_threshold: float = SLOW_THRESHOLD,
) -> dict[str, float]:
    """Compute regression metrics: MAE, RMSE, R2, tolerance accuracy, and slow detection."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "tolerance_accuracy": 0.0,
            "slow_precision": 0.0,
            "slow_recall": 0.0,
            "slow_f1": 0.0,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = 0.0

    # Tolerance accuracy: fraction within +/- 0.25 of true ratio
    tol_acc = float(np.mean(np.abs(y_true - y_pred) <= 0.25))

    # Slow detection at threshold (e.g. 1.5)
    y_true_slow = (y_true >= slow_threshold).astype(int)
    y_pred_slow = (y_pred >= slow_threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_slow, y_pred_slow, average="binary", zero_division=0
    )

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "tolerance_accuracy": round(tol_acc, 4),
        "slow_precision": round(float(p), 4),
        "slow_recall": round(float(r), 4),
        "slow_f1": round(float(f1), 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Compute multi-class classification metrics, per-class stats, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "classes": classes,
        }

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    m_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per class metrics
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class = {
        cls: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(classes, prec, rec, f1, supp, strict=False)
    }

    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(m_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classes": classes,
    }


def compute_anomaly_metrics(
    scores: np.ndarray,
    y_true_binary: np.ndarray,
    threshold: float,
    is_combo_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute anomaly metrics: ROC-AUC, recall at threshold, false alarm rate, and combo rate."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true_binary = np.asarray(y_true_binary, dtype=np.int32)

    preds = (scores >= threshold).astype(int)

    # ROC-AUC if both classes present
    if len(np.unique(y_true_binary)) > 1:
        roc = float(roc_auc_score(y_true_binary, scores))
    else:
        roc = float("nan")

    # Recall on anomalous rows (y_true_binary == 1)
    anom_mask = y_true_binary == 1
    recall = float(np.mean(preds[anom_mask])) if np.sum(anom_mask) > 0 else 0.0

    # False alarm rate on healthy rows (y_true_binary == 0)
    healthy_mask = y_true_binary == 0
    far = float(np.mean(preds[healthy_mask])) if np.sum(healthy_mask) > 0 else 0.0

    # Combo detection rate
    if is_combo_mask is not None and np.sum(is_combo_mask) > 0:
        combo_rate = float(np.mean(preds[is_combo_mask]))
    else:
        combo_rate = float("nan")

    return {
        "roc_auc": round(roc, 4),
        "recall": round(recall, 4),
        "false_alarm_rate": round(far, 4),
        "combo_detection_rate": round(combo_rate, 4),
        "threshold": round(threshold, 4),
    }


def compute_breakdowns(
    df: Any,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    y_true_clf: np.ndarray | None = None,
    y_pred_clf: np.ndarray | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Breakdown MAE and macro F1 by scenario, intensity, and pattern."""
    breakdowns: dict[str, dict[str, dict[str, float]]] = {
        "scenario": {},
        "intensity": {},
        "pattern": {},
    }

    for col in ["scenario", "intensity", "pattern"]:
        if col not in df.columns:
            continue
        unique_vals = sorted(df[col].dropna().unique())
        for val in unique_vals:
            mask = (df[col] == val).to_numpy()
            if not np.any(mask):
                continue
            sub_y_reg = y_true_reg[mask]
            sub_p_reg = y_pred_reg[mask]
            mae = float(np.mean(np.abs(sub_y_reg - sub_p_reg)))

            f1 = float("nan")
            if y_true_clf is not None and y_pred_clf is not None:
                sub_y_clf = y_true_clf[mask]
                sub_p_clf = y_pred_clf[mask]
                if len(sub_y_clf) > 0:
                    f1 = float(f1_score(sub_y_clf, sub_p_clf, average="macro", zero_division=0))

            breakdowns[col][str(val)] = {
                "count": float(np.sum(mask)),
                "mae": round(mae, 4),
                "macro_f1": round(f1, 4),
            }

    return breakdowns


def compute_block_bootstrap(
    run_ids: np.ndarray,
    y_true_reg: np.ndarray,
    preds_reg: dict[str, np.ndarray],
    y_true_clf: np.ndarray | None = None,
    preds_clf: dict[str, np.ndarray] | None = None,
    n_resamples: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute 95% confidence intervals via block bootstrap resampled by run_id."""
    rng = np.random.RandomState(seed)
    unique_runs = np.unique(run_ids)
    n_runs = len(unique_runs)

    if n_runs < 2:
        return {}

    # Pre-index indices per run
    run_to_indices = {r: np.where(run_ids == r)[0] for r in unique_runs}

    reg_mae_samples: dict[str, list[float]] = {m: [] for m in preds_reg}
    reg_r2_samples: dict[str, list[float]] = {m: [] for m in preds_reg}
    persistence_mae_samples: list[float] = []

    clf_f1_samples: dict[str, list[float]] = {m: [] for m in (preds_clf or {})}

    for _ in range(n_resamples):
        sampled_runs = rng.choice(unique_runs, size=n_runs, replace=True)
        sample_indices = np.concatenate([run_to_indices[r] for r in sampled_runs])

        y_tr_sub = y_true_reg[sample_indices]

        # Regressor bootstrap samples
        p_pers = preds_reg.get("Persistence")
        pers_mae = (
            float(np.mean(np.abs(y_tr_sub - p_pers[sample_indices]))) if p_pers is not None else 0.0
        )
        persistence_mae_samples.append(pers_mae)

        for m_name, preds in preds_reg.items():
            p_sub = preds[sample_indices]
            mae = float(np.mean(np.abs(y_tr_sub - p_sub)))
            try:
                r2 = float(r2_score(y_tr_sub, p_sub))
            except Exception:
                r2 = 0.0
            reg_mae_samples[m_name].append(mae)
            reg_r2_samples[m_name].append(r2)

        # Classifier bootstrap samples
        if y_true_clf is not None and preds_clf:
            y_cl_sub = y_true_clf[sample_indices]
            for m_name, preds in preds_clf.items():
                p_sub = preds[sample_indices]
                f1 = float(f1_score(y_cl_sub, p_sub, average="macro", zero_division=0))
                clf_f1_samples[m_name].append(f1)

    # Compute percentiles
    results: dict[str, Any] = {"regression": {}, "classification": {}}

    for m_name in preds_reg:
        maes = np.array(reg_mae_samples[m_name])
        r2s = np.array(reg_r2_samples[m_name])

        mae_ci = [
            round(float(np.percentile(maes, 2.5)), 4),
            round(float(np.percentile(maes, 97.5)), 4),
        ]
        r2_ci = [
            round(float(np.percentile(r2s, 2.5)), 4),
            round(float(np.percentile(r2s, 97.5)), 4),
        ]

        # Improvement over Persistence (persistence_mae - model_mae)
        pers_diff = np.array(persistence_mae_samples) - maes
        diff_ci = [
            round(float(np.percentile(pers_diff, 2.5)), 4),
            round(float(np.percentile(pers_diff, 97.5)), 4),
        ]
        better = bool(diff_ci[0] > 0.0)

        results["regression"][m_name] = {
            "mae_ci": mae_ci,
            "r2_ci": r2_ci,
            "improvement_over_persistence_ci": diff_ci,
            "better_than_persistence": better,
        }

    for m_name, f1_list in clf_f1_samples.items():
        f1s = np.array(f1_list)
        f1_ci = [
            round(float(np.percentile(f1s, 2.5)), 4),
            round(float(np.percentile(f1s, 97.5)), 4),
        ]
        results["classification"][m_name] = {"macro_f1_ci": f1_ci}

    return results
