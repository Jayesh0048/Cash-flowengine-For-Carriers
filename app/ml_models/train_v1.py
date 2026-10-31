"""
train_v2.py -- Leak-aware, residual-boosted weekly cash-flow engine.

Predicts weekly cash-impact/gross ratio relative to a 3-week rolling baseline.

All model features use information from prior weeks. Dollar evaluation uses
the actual scheduled gross from the target week to properly calculate MAE.
"""

from __future__ import annotations

import os
import argparse
import json
from datetime import datetime, timezone
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved")
MODEL_FILE = os.path.join(MODEL_DIR, "lgbm_cashflow_v2.pkl")


def model_file_for(account_id: str) -> str:
    """Return the portfolio or account-specific model artifact location."""
    if account_id == "all":
        return MODEL_FILE
    return os.path.join(MODEL_DIR, f"lgbm_cashflow_v2_{account_id}.pkl")


def report_file_for(account_id: str) -> str:
    """Return the human-readable validation report path for a model run."""
    suffix = "" if account_id == "all" else f"_{account_id}"
    return os.path.join(MODEL_DIR, f"lgbm_cashflow_v2{suffix}.report.json")

FREIGHT_SOURCE_TYPES = {"LOAD"}

ACTIVE_MIN_GROSS = 2_000.0

N_SPLITS = 5
TEST_WEEKS = 8
MINIMUM_TRAINING_WEEKS = 40

# Require model to be strictly better than or equal to baseline USD MAE
MIN_USD_MAE_IMPROVEMENT = 0.00


# Every model feature must be available at forecast time.
FEATURES = [
    "last_active_gross",
    "last_active_loads",
    "last_active_avg_rate_per_load",
    "lag_1w_inflow_ratio",
    "lag_1w_active",
    "rolling_3w_ratio_mean",
]

FORBIDDEN = {
    "target_ratio",
    "cash_impact",
    "inflow",
    "outflow",
    "gross",
    "loads",
    "inflow_ratio",
}

assert FORBIDDEN.isdisjoint(FEATURES), (
    "Target leakage detected in feature schema!"
)


# ============================================================
# DATA EXTRACTION
# ============================================================

def load_weekly_from_db(account_id: str = "all") -> pd.DataFrame:
    """Load settled cash events for one account or the portfolio.

    A training target is only valid after its cash event has settled.  Pending
    events are forecasts, not observed outcomes, and including them would make
    model validation overstate performance.
    """

    from app.core.database import SessionLocal
    from app.core.models import CashEvent

    db = SessionLocal()

    try:
        query = db.query(CashEvent).filter(
            CashEvent.due_date.isnot(None),
            CashEvent.status == "SETTLED",
        )
        if account_id != "all":
            query = query.filter(CashEvent.account_id == account_id)
        events = query.all()
    finally:
        db.close()

    if not events:
        raise ValueError(
            f"No settled cash events were found for account '{account_id}'."
        )

    rows = []

    for event in events:
        net_amount = float(event.net_amount or 0.0)

        direction = str(event.direction or "").upper()
        source_type = str(event.source_type or "").upper()

        is_freight = source_type in FREIGHT_SOURCE_TYPES

        rows.append(
            {
                "timestamp": event.due_date,

                "inflow": (
                    net_amount
                    if direction == "INFLOW"
                    else 0.0
                ),

                "outflow": (
                    abs(net_amount)
                    if direction == "OUTFLOW"
                    else 0.0
                ),

                "gross": (
                    float(event.gross_amount or 0.0)
                    if is_freight
                    else 0.0
                ),

                "loads": 1 if is_freight else 0,
            }
        )

    return to_weekly(pd.DataFrame(rows))


def data_quality_report(account_id: str = "all") -> dict[str, Any]:
    """Summarize ledger suitability before training without changing data."""
    from app.core.database import SessionLocal
    from app.core.models import CashEvent

    db = SessionLocal()
    try:
        query = db.query(CashEvent)
        if account_id != "all":
            query = query.filter(CashEvent.account_id == account_id)
        events = query.all()
    finally:
        db.close()

    dated_events = [event for event in events if event.due_date is not None]
    settled_events = [event for event in dated_events if event.status == "SETTLED"]
    active_loads = [
        event for event in settled_events
        if str(event.source_type or "").upper() in FREIGHT_SOURCE_TYPES
        and float(event.gross_amount or 0.0) >= ACTIVE_MIN_GROSS
    ]
    invalid_amounts = [
        event.event_id for event in settled_events
        if not np.isfinite(float(event.net_amount or 0.0))
        or not np.isfinite(float(event.gross_amount or 0.0))
    ]
    dates = [event.due_date for event in settled_events]

    return {
        "account_id": account_id,
        "events_total": len(events),
        "events_with_due_date": len(dated_events),
        "settled_events": len(settled_events),
        "pending_or_other_events": len(dated_events) - len(settled_events),
        "eligible_settled_load_events": len(active_loads),
        "invalid_numeric_event_ids": invalid_amounts,
        "settled_date_range": {
            "start": min(dates).isoformat() if dates else None,
            "end": max(dates).isoformat() if dates else None,
        },
    }


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

def to_weekly(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level cash data into Sunday-ending weeks."""

    required = {
        "timestamp",
        "inflow",
        "outflow",
        "gross",
        "loads",
    }

    missing = required.difference(events.columns)

    if missing:
        raise ValueError(
            f"Event data is missing required columns: "
            f"{sorted(missing)}"
        )

    if events.empty:
        raise ValueError(
            "Cannot aggregate an empty event DataFrame."
        )

    df = events.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        raise ValueError(
            "No events have a valid timestamp."
        )

    for column in (
        "inflow",
        "outflow",
        "gross",
        "loads",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0.0)

    df = (
        df.sort_values("timestamp")
        .set_index("timestamp")
    )

    weekly = (
        df.resample("W-SUN")
        .agg(
            {
                "inflow": "sum",
                "outflow": "sum",
                "gross": "sum",
                "loads": "sum",
            }
        )
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    # Keep timestamps consistently timezone-naive in UTC.
    if weekly["date"].dt.tz is not None:
        weekly["date"] = (
            weekly["date"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
        )

    weekly["cash_impact"] = (
        weekly["inflow"] - weekly["outflow"]
    )

    return (
        weekly.sort_values("date")
        .reset_index(drop=True)
    )


# ============================================================
# REMOVE INCOMPLETE WEEKS
# ============================================================

def drop_incomplete_weeks(
    weekly: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    """
    Remove the current incomplete week.

    Weekly date labels are Sunday-ending week labels represented
    as timezone-naive UTC timestamps.
    """

    if weekly.empty:
        raise ValueError(
            "Weekly data is empty before incomplete-week filtering."
        )

    now_value = now or datetime.now(timezone.utc)

    timestamp = pd.Timestamp(now_value)

    if timestamp.tzinfo is None:
        cutoff = timestamp.normalize()
    else:
        cutoff = (
            timestamp.tz_convert("UTC")
            .tz_localize(None)
            .normalize()
        )

    result = weekly.loc[
        weekly["date"] < cutoff
    ].copy()

    result = result.reset_index(drop=True)

    if result.empty:
        raise ValueError(
            "No completed weeks remain after filtering "
            "the current week."
        )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build causal features and the cash-yield target.
    """

    required = {
        "date",
        "inflow",
        "outflow",
        "gross",
        "loads",
        "cash_impact",
    }

    missing = required.difference(weekly.columns)

    if missing:
        raise ValueError(
            f"Weekly data is missing required columns: "
            f"{sorted(missing)}"
        )

    if weekly.empty:
        raise ValueError(
            "Cannot build features from empty weekly data."
        )

    d = (
        weekly.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    d["date"] = pd.to_datetime(
        d["date"],
        errors="coerce",
    )

    d = (
        d.dropna(subset=["date"])
        .reset_index(drop=True)
    )

    for column in (
        "inflow",
        "outflow",
        "gross",
        "loads",
        "cash_impact",
    ):
        d[column] = pd.to_numeric(
            d[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # 1. Active operating weeks
    # --------------------------------------------------------

    d["active"] = (
        d["gross"].ge(ACTIVE_MIN_GROSS)
        & d["gross"].gt(0.0)
    )

    # --------------------------------------------------------
    # 2. Target ratio
    # --------------------------------------------------------

    d["target_ratio"] = (
        d["cash_impact"]
        / d["gross"].where(d["gross"].gt(0.0))
    ).where(d["active"])

    d["inflow_ratio"] = (
        d["inflow"]
        / d["gross"].where(d["gross"].gt(0.0))
    ).where(d["active"])

    # --------------------------------------------------------
    # 3. Three-calendar-week rolling baseline
    # --------------------------------------------------------

    d["rolling_3w_ratio_mean"] = (
        d["target_ratio"]
        .shift(1)
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    # --------------------------------------------------------
    # 4. Previous active operating-week inputs
    # --------------------------------------------------------

    d["last_active_gross"] = (
        d["gross"]
        .where(d["active"])
        .ffill()
        .shift(1)
        .fillna(0.0)
    )

    d["last_active_loads"] = (
        d["loads"]
        .where(d["active"])
        .ffill()
        .shift(1)
        .fillna(0.0)
    )

    avg_rate = (
        d["gross"]
        / d["loads"].clip(lower=1.0)
    )

    d["last_active_avg_rate_per_load"] = (
        avg_rate
        .where(d["active"])
        .ffill()
        .shift(1)
        .fillna(0.0)
    )

    d["lag_1w_inflow_ratio"] = (
        d["inflow_ratio"]
        .shift(1)
        .fillna(0.0)
    )

    # Safely handle the boolean shift to prevent Pandas FutureWarnings
    d["lag_1w_active"] = (
        d["active"]
        .shift(1)
        .astype(float)
        .fillna(0.0)
        .astype("int8")
    )

    # Evaluate USD accuracy directly against the target week's actual gross
    d["forecast_gross_proxy"] = d["gross"]

    # Fallback baseline estimate for the next forecast.
    recent_calendar_targets = (
        d["target_ratio"]
        .tail(3)
        .dropna()
    )

    if not recent_calendar_targets.empty:
        default_baseline = float(
            recent_calendar_targets.mean()
        )
    else:
        default_baseline = 0.0

    # --------------------------------------------------------
    # 5. Final training rows
    # --------------------------------------------------------

    result = d.loc[
        d["active"]
        & d["rolling_3w_ratio_mean"].notna()
    ].copy()

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if result.empty:
        raise ValueError(
            "No usable active weeks remain after feature construction. "
            "Check gross amounts and ACTIVE_MIN_GROSS."
        )

    check_columns = FEATURES + [
        "target_ratio",
        "cash_impact",
        "forecast_gross_proxy",
    ]

    missing_counts = (
        result[check_columns]
        .isna()
        .sum()
    )

    bad_columns = missing_counts[
        missing_counts > 0
    ].to_dict()

    if bad_columns:
        raise ValueError(
            "Missing/non-finite values in training data: "
            f"{bad_columns}"
        )

    result = result.reset_index(drop=True)

    result.attrs["default_baseline_ratio"] = (
        default_baseline
    )

    return result


# ============================================================
# LIGHTGBM MODEL
# ============================================================

def make_residual_lgbm() -> lgb.LGBMRegressor:
    """Build a small, regularized model for baseline residuals."""

    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=60,
        learning_rate=0.03,
        num_leaves=5,
        max_depth=2,
        min_child_samples=4,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


# ============================================================
# CHRONOLOGICAL VALIDATION PLAN
# ============================================================

def split_plan(
    n: int,
) -> tuple[int, int]:
    """Create expanding-window time-series validation folds."""

    if n < MINIMUM_TRAINING_WEEKS:
        raise ValueError(
            f"Only {n} usable active weeks found. "
            f"At least {MINIMUM_TRAINING_WEEKS} "
            "are required for this validation plan."
        )

    test_size = max(
        4,
        min(TEST_WEEKS, n // 10),
    )

    n_splits = min(
        N_SPLITS,
        (n - 30) // test_size,
    )

    if n_splits < 2:
        raise ValueError(
            f"Cannot create at least two chronological "
            f"validation folds from {n} weeks."
        )

    return n_splits, test_size


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_residual(
    data: pd.DataFrame,
    n_splits: int,
    test_size: int,
) -> dict[str, np.ndarray]:
    """
    Produce out-of-fold predictions using chronological splits.
    """

    X = data[FEATURES].astype(float)

    y = data["target_ratio"].to_numpy(
        dtype=float
    )

    baseline = data[
        "rolling_3w_ratio_mean"
    ].to_numpy(dtype=float)

    n = len(data)

    oof = {
        "base_mean": np.full(n, np.nan),
        "base_roll3": np.full(n, np.nan),
        "hybrid_lgbm": np.full(n, np.nan),
    }

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
    )

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y[train_indices]

        baseline_train = baseline[train_indices]
        baseline_test = baseline[test_indices]

        residual_train = (
            y_train - baseline_train
        )

        train_mean = float(
            np.mean(y_train)
        )

        oof["base_mean"][test_indices] = (
            train_mean
        )

        oof["base_roll3"][test_indices] = (
            baseline_test
        )

        model = make_residual_lgbm()

        model.fit(
            X_train,
            residual_train,
        )

        residual_prediction = model.predict(
            X_test
        )

        oof["hybrid_lgbm"][test_indices] = (
            baseline_test + residual_prediction
        )

        print(
            f"    Fold {fold_number}/{n_splits}: "
            f"train={len(train_indices)} observations, "
            f"test={len(test_indices)} observations, "
            f"train-through="
            f"{data.iloc[train_indices[-1]]['date'].date()}, "
            f"test-start="
            f"{data.iloc[test_indices[0]]['date'].date()}"
        )

    return oof


# ============================================================
# METRIC HELPERS
# ============================================================

def _safe_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Return NaN if R² is mathematically undefined."""

    if len(actual) < 2:
        return float("nan")

    if np.allclose(actual, actual[0]):
        return float("nan")

    return float(
        r2_score(actual, predicted)
    )


# ============================================================
# EVALUATION AND MODEL BUNDLE
# ============================================================

def evaluate_and_save(
    data: pd.DataFrame,
    save: bool = True,
    account_id: str = "all",
) -> dict[str, Any]:
    """
    Evaluate baselines versus hybrid, select the champion,
    train the final model, and optionally save the bundle.
    """

    n = len(data)

    if n == 0:
        raise ValueError(
            "Training DataFrame is empty."
        )

    print(
        f"[*] Usable active weeks: {n} "
        f"({data['date'].min().date()} -> "
        f"{data['date'].max().date()})"
    )

    n_splits, test_size = split_plan(n)

    print(
        f"[*] Walk-forward validation: "
        f"{n_splits} folds x {test_size} observations "
        "(chronological, no shuffling)"
    )

    oof = walk_forward_residual(
        data,
        n_splits,
        test_size,
    )

    y_ratio = data[
        "target_ratio"
    ].to_numpy(dtype=float)

    actual_usd = data[
        "cash_impact"
    ].to_numpy(dtype=float)

    forecast_gross = data[
        "forecast_gross_proxy"
    ].to_numpy(dtype=float)

    mask = np.isfinite(
        oof["hybrid_lgbm"]
    )

    if int(mask.sum()) < 2:
        raise ValueError(
            "Not enough out-of-fold predictions to evaluate."
        )

    labels = {
        "base_mean": "Baseline: Historical Mean",
        "base_roll3": "Baseline: 3-Week Rolling Ratio",
        "hybrid_lgbm": "Residual LightGBM Engine (v2)",
    }

    scores: dict[str, dict[str, Any]] = {}

    for name, label in labels.items():
        predicted_ratio = oof[name][mask]

        predicted_usd = (
            predicted_ratio
            * forecast_gross[mask]
        )

        actual_usd_masked = actual_usd[mask]

        scores[name] = {
            "label": label,

            "ratio_mae": float(
                mean_absolute_error(
                    y_ratio[mask],
                    predicted_ratio,
                )
            ),

            "usd_mae": float(
                mean_absolute_error(
                    actual_usd_masked,
                    predicted_usd,
                )
            ),

            "usd_rmse": float(
                np.sqrt(
                    mean_squared_error(
                        actual_usd_masked,
                        predicted_usd,
                    )
                )
            ),

            "usd_r2": _safe_r2(
                actual_usd_masked,
                predicted_usd,
            ),
        }

    base_ratio_mae = scores[
        "base_roll3"
    ]["ratio_mae"]

    hybrid_ratio_mae = scores[
        "hybrid_lgbm"
    ]["ratio_mae"]

    base_usd_mae = scores[
        "base_roll3"
    ]["usd_mae"]

    hybrid_usd_mae = scores[
        "hybrid_lgbm"
    ]["usd_mae"]

    usd_improvement = (
        base_usd_mae - hybrid_usd_mae
    )

    usd_improvement_pct = (
        usd_improvement / base_usd_mae
        if base_usd_mae > 0.0
        else 0.0
    )

    # --------------------------------------------------------
    # Print evaluation table
    # --------------------------------------------------------

    print(
        f"\n[*] Out-of-sample results across "
        f"{int(mask.sum())} evaluation observations:"
    )

    print("    " + "-" * 100)

    print(
        f"    {'Model / Strategy':<34}"
        f"{'Ratio MAE':>12}"
        f"{'USD MAE':>15}"
        f"{'USD RMSE':>15}"
        f"{'USD R²':>10}"
    )

    print("    " + "-" * 100)

    for name, score in scores.items():
        r2_value = score["usd_r2"]

        r2_text = (
            f"{r2_value:.3f}"
            if np.isfinite(r2_value)
            else "n/a"
        )

        benchmark_note = (
            " <- benchmark"
            if name == "base_roll3"
            else ""
        )

        print(
            f"    {score['label']:<34}"
            f"{score['ratio_mae'] * 100:>10.2f}pt  "
            f"${score['usd_mae']:>13,.0f}"
            f"${score['usd_rmse']:>13,.0f}"
            f"{r2_text:>10}"
            f"{benchmark_note}"
        )

    print("    " + "-" * 100)

    # --------------------------------------------------------
    # Empirical 80% residual error band
    # --------------------------------------------------------

    residuals_oof = (
        y_ratio[mask]
        - oof["hybrid_lgbm"][mask]
    )

    # Use inclusive order statistics instead of interpolated percentiles.  With
    # a small validation sample, interpolation can produce an interval labelled
    # 80% that covers fewer than 80% of the actual out-of-fold observations.
    ordered_residuals = np.sort(residuals_oof)
    lower_rank = int(np.floor(len(ordered_residuals) * 0.10))
    upper_rank = int(np.ceil(len(ordered_residuals) * 0.90)) - 1
    residual_band_80 = {
        "low": float(ordered_residuals[lower_rank]),
        "high": float(ordered_residuals[upper_rank]),
    }

    coverage = float(
        np.mean(
            (
                residuals_oof
                >= residual_band_80["low"]
            )
            & (
                residuals_oof
                <= residual_band_80["high"]
            )
        )
    )

    print(
        "\n[*] Empirical 10th-90th percentile "
        "out-of-fold error band: "
        f"[{residual_band_80['low'] * 100:+.2f}pt, "
        f"{residual_band_80['high'] * 100:+.2f}pt]; "
        f"observed inclusion={coverage * 100:.1f}%."
    )

    # --------------------------------------------------------
    # Champion selection
    # --------------------------------------------------------

    approved = bool(
        hybrid_usd_mae
        <= base_usd_mae * (
            1.0 - MIN_USD_MAE_IMPROVEMENT
        )
        and hybrid_ratio_mae < base_ratio_mae
    )

    champion = (
        "hybrid_lgbm"
        if approved
        else "base_roll3"
    )

    print(
        "\n============================================================="
    )

    if approved:
        print(
            "  VERDICT: Hybrid passes the configured validation rule. "
            f"USD MAE reduction=${usd_improvement:,.2f} "
            f"({usd_improvement_pct * 100:.1f}%)."
        )

        print(
            "  CHAMPION: hybrid_lgbm"
        )

    else:
        print(
            "  VERDICT: Hybrid did not meet both approval criteria."
        )

        print(
            "  CHAMPION: base_roll3"
        )

    print(
        "=============================================================\n"
    )

    # --------------------------------------------------------
    # Fit final model on all available historical rows
    # --------------------------------------------------------

    X_all = data[
        FEATURES
    ].astype(float)

    y_all = data[
        "target_ratio"
    ].to_numpy(dtype=float)

    baseline_all = data[
        "rolling_3w_ratio_mean"
    ].to_numpy(dtype=float)

    residuals_all = (
        y_all - baseline_all
    )

    final_model = make_residual_lgbm()

    final_model.fit(
        X_all,
        residuals_all,
    )

    default_baseline = data.attrs.get(
        "default_baseline_ratio"
    )

    if (
        default_baseline is None
        or not np.isfinite(default_baseline)
    ):
        recent_targets = (
            data["target_ratio"]
            .tail(3)
            .dropna()
        )

        default_baseline = (
            float(recent_targets.mean())
            if not recent_targets.empty
            else 0.0
        )

    # --------------------------------------------------------
    # Model bundle
    # --------------------------------------------------------

    bundle: dict[str, Any] = {
        "model": final_model,

        "champion": champion,

        "features": FEATURES,

        "residual_band_80": residual_band_80,

        "residual_band_observed_coverage": coverage,

        "default_baseline_ratio": float(
            default_baseline
        ),

        "model_type": "residual_boosted_ratio_v2",

        "approved": approved,

        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "n_weeks": int(n),

        "account_id": account_id,

        "training_data_policy": "settled_cash_events_only",

        "metrics": scores,

        "validation_observations": int(mask.sum()),
    }

    # --------------------------------------------------------
    # Save bundle
    # --------------------------------------------------------

    if save:
        os.makedirs(
            MODEL_DIR,
            exist_ok=True,
        )

        joblib.dump(
            bundle,
            model_file_for(account_id),
        )

        report = {
            "account_id": account_id,
            "trained_at": bundle["trained_at"],
            "approved": approved,
            "champion": champion,
            "training_weeks": int(n),
            "validation_observations": int(mask.sum()),
            "metrics": scores,
            "residual_band_80": residual_band_80,
            "residual_band_observed_coverage": coverage,
            "data_quality": data_quality_report(account_id),
        }
        with open(report_file_for(account_id), "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, sort_keys=True)

        print(
            f"[✓] Model bundle saved to {model_file_for(account_id)} "
            f"(hybrid_approved={approved}, "
            f"champion={champion})"
        )
        print(f"[✓] Validation report saved to {report_file_for(account_id)}")

    return bundle


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leak-aware cash-flow model from settled ledger events."
    )
    parser.add_argument(
        "--account-id",
        choices=("all", "acc1", "acc2"),
        default="all",
        help="Train a portfolio model or an account-specific model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate the model without replacing a saved artifact.",
    )
    args = parser.parse_args()

    quality = data_quality_report(args.account_id)
    print("[*] Training data quality:")
    print(json.dumps(quality, indent=2))
    if quality["invalid_numeric_event_ids"]:
        raise ValueError("Training stopped: invalid numeric cash-event values found.")

    print(
        f"[*] Connecting to database for '{args.account_id}' training corpus..."
    )

    weekly = drop_incomplete_weeks(
        load_weekly_from_db(args.account_id)
    )

    data = build_features(
        weekly
    )

    evaluate_and_save(
        data,
        save=not args.dry_run,
        account_id=args.account_id,
    )


if __name__ == "__main__":
    main()
