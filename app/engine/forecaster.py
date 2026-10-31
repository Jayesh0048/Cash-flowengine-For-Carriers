import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.models import CashEvent

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "ml_models", "saved", "lgbm_cashflow_v2.pkl"
)


def account_model_path(account_id: str) -> str:
    """Return the optional account-specific model artifact path."""
    return os.path.join(
        os.path.dirname(__file__), "..", "ml_models", "saved",
        f"lgbm_cashflow_v2_{account_id}.pkl",
    )

# Account-specific liquid balance parameters
ACCOUNT_PROFILES: Dict[str, Dict[str, float]] = {
    "all": {"starting_cash": 45000.0, "safety_floor": 25000.0},
    "acc1": {"starting_cash": 25000.0, "safety_floor": 15000.0},
    "acc2": {"starting_cash": 20000.0, "safety_floor": 10000.0},
}


class LiquidityForecaster:
    """Multi-fleet liquidity forecasting engine utilizing Residual LightGBM v2.

    Predicts net freight conversion yields and models 30/60/90-day cash horizons
    accounting for factoring terms, fuel/maintenance burn, and empirical error bands.
    """

    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.account_model_bundles: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(MODEL_PATH):
            try:
                self.model_bundle = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"[!] Warning: Could not load ML model bundle: {e}")
        for account_id in ("acc1", "acc2"):
            path = account_model_path(account_id)
            if os.path.exists(path):
                try:
                    self.account_model_bundles[account_id] = joblib.load(path)
                except Exception as e:
                    print(f"[!] Warning: Could not load {account_id} model bundle: {e}")

    def _model_for_account(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Prefer a validated account model; otherwise use a validated portfolio model."""
        account_model = self.account_model_bundles.get(account_id)
        if account_model and account_model.get("approved", False):
            return account_model
        if self.model_bundle and self.model_bundle.get("approved", False):
            return self.model_bundle
        return None

    def _get_account_historical_ratio(
        self, session: Session, account_id: Optional[str] = None
    ) -> float:
        """Computes the recent 3-week rolling baseline conversion ratio from settled records."""
        query = session.query(CashEvent).filter(
            CashEvent.due_date.isnot(None),
            CashEvent.status == "SETTLED",
        )
        if account_id and account_id != "all":
            query = query.filter(CashEvent.account_id == account_id)

        events = query.all()
        if not events:
            return (
                float(self.model_bundle.get("default_baseline_ratio", 0.82))
                if self.model_bundle
                else 0.82
            )

        records = []
        for e in events:
            net = float(e.net_amount or 0.0)
            impact = net if e.direction == "INFLOW" else -abs(net)
            records.append({
                "timestamp": e.due_date,
                "impact": impact,
                "gross": float(e.gross_amount or 0.0) if e.source_type == "LOAD" else 0.0,
            })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        weekly = df.set_index("timestamp").resample("W-SUN").sum().reset_index()

        active = weekly[weekly["gross"] >= 1000.0].tail(3)
        if len(active) == 0:
            return (
                float(self.model_bundle.get("default_baseline_ratio", 0.82))
                if self.model_bundle
                else 0.82
            )

        ratio = active["impact"].sum() / np.maximum(active["gross"].sum(), 1000.0)
        return float(np.clip(ratio, 0.50, 0.95))

    def _get_recent_lag_metrics(
        self, session: Session, account_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculates lag-1 week inflow ratio and activity status for ML feature alignment."""
        query = session.query(CashEvent).filter(
            CashEvent.due_date.isnot(None),
            CashEvent.status == "SETTLED",
        )
        if account_id and account_id != "all":
            query = query.filter(CashEvent.account_id == account_id)

        events = query.all()
        if not events:
            return {"lag_1w_inflow_ratio": 0.85, "lag_1w_active": 1.0}

        records = []
        for e in events:
            net = float(e.net_amount or 0.0)
            records.append({
                "timestamp": e.due_date,
                "inflow": net if e.direction == "INFLOW" else 0.0,
                "gross": float(e.gross_amount or 0.0) if e.source_type == "LOAD" else 0.0,
            })

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        weekly = df.set_index("timestamp").resample("W-SUN").sum().reset_index()

        if len(weekly) < 2:
            return {"lag_1w_inflow_ratio": 0.85, "lag_1w_active": 1.0}

        last_week = weekly.iloc[-1]
        gross_val = float(last_week["gross"])
        inflow_val = float(last_week["inflow"])
        is_active = 1.0 if gross_val >= 2000.0 else 0.0
        ratio = (inflow_val / gross_val) if gross_val > 0.0 else 0.85

        return {
            "lag_1w_inflow_ratio": float(np.clip(ratio, 0.50, 0.95)),
            "lag_1w_active": is_active,
        }

    def calculate_forecast(
        self,
        days: int = 30,
        account_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Core multi-horizon forecasting logic (30, 60, 90+ days)."""
        session = db if db is not None else (self.db or SessionLocal())
        should_close = db is None and self.db is None

        try:
            acc_key = (account_id or "all").lower()
            profile = ACCOUNT_PROFILES.get(acc_key, ACCOUNT_PROFILES["all"])
            model_bundle = self._model_for_account(acc_key)

            # Use naive UTC timestamps for clean SQLite string/date comparisons
            now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            target_horizon = now_naive + timedelta(days=days)

            # 1. Upcoming Scheduled Events in Horizon
            base_query = session.query(CashEvent).filter(
                CashEvent.due_date >= now_naive,
                CashEvent.due_date <= target_horizon,
            )
            if acc_key != "all":
                base_query = base_query.filter(CashEvent.account_id == acc_key)

            upcoming_events = base_query.all()

            firm_inflows = sum(
                float(e.net_amount or 0.0)
                for e in upcoming_events
                if e.direction == "INFLOW" and e.net_amount
            )
            firm_outflows = sum(
                float(e.net_amount or 0.0)
                for e in upcoming_events
                if e.direction == "OUTFLOW" and e.net_amount
            )
            scheduled_gross = sum(
                float(e.gross_amount or 0.0)
                for e in upcoming_events
                if e.source_type == "LOAD" and e.gross_amount
            )
            scheduled_loads = sum(
                1 for e in upcoming_events if e.source_type == "LOAD"
            )

            # Isolate predictive maintenance work orders from generic operating bills
            scheduled_maintenance_outflow = sum(
                float(e.net_amount or 0.0)
                for e in upcoming_events
                if e.direction == "OUTFLOW" and e.source_type == "MAINTENANCE"
            )
            non_maintenance_outflows = max(0.0, firm_outflows - scheduled_maintenance_outflow)

            # 2. Historical Baseline Monthly Pace
            past_query = session.query(CashEvent).filter(
                CashEvent.due_date < now_naive,
                CashEvent.direction == "INFLOW",
                CashEvent.status == "SETTLED",
            )
            if acc_key != "all":
                past_query = past_query.filter(CashEvent.account_id == acc_key)

            past_inflows = past_query.all()
            if past_inflows:
                earliest_raw = min(e.due_date for e in past_inflows if e.due_date)
                earliest_naive = (
                    earliest_raw.replace(tzinfo=None)
                    if hasattr(earliest_raw, "tzinfo") and earliest_raw.tzinfo
                    else earliest_raw
                )
                total_past_inflow = sum(float(e.net_amount or 0.0) for e in past_inflows)
                elapsed_days = max(30, (now_naive - earliest_naive).days)
                elapsed_months = elapsed_days / 30.4
                baseline_monthly_inflow = total_past_inflow / max(elapsed_months, 1.0)
            else:
                baseline_monthly_inflow = 120000.0 if acc_key == "acc2" else 150000.0

            month_factor = max(1.0, days / 30.0)
            rolling_baseline = self._get_account_historical_ratio(session, acc_key)
            predicted_ratio = rolling_baseline
            band_low, band_high = -0.039, 0.053

            # 3. Dynamic Residual LightGBM v2 Inference
            if model_bundle and model_bundle.get("approved", False):
                try:
                    model = model_bundle["model"]
                    features = model_bundle["features"]
                    band = model_bundle.get("residual_band_80", {})
                    band_low = band.get("low", -0.039)
                    band_high = band.get("high", 0.053)

                    gross_input = (
                        scheduled_gross
                        if scheduled_gross > 0
                        else (baseline_monthly_inflow * month_factor)
                    )
                    loads_input = max(scheduled_loads, int(month_factor * 15))
                    avg_rate = gross_input / max(loads_input, 1)

                    lag_metrics = self._get_recent_lag_metrics(session, acc_key)

                    inference_df = pd.DataFrame([{
                        "last_active_gross": gross_input,
                        "last_active_loads": loads_input,
                        "last_active_avg_rate_per_load": avg_rate,
                        "lag_1w_inflow_ratio": lag_metrics["lag_1w_inflow_ratio"],
                        "lag_1w_active": lag_metrics["lag_1w_active"],
                        "rolling_3w_ratio_mean": rolling_baseline,
                    }])[features]

                    predicted_delta = float(model.predict(inference_df)[0])
                    predicted_ratio = float(
                        np.clip(rolling_baseline + predicted_delta, 0.50, 0.95)
                    )
                except Exception as err:
                    print(f"[!] Live inference error: {err}")
                    predicted_ratio = rolling_baseline

            # 4. Projected Inflows
            expected_run_rate_inflows = (baseline_monthly_inflow * month_factor) * (
                predicted_ratio / max(rolling_baseline, 0.01)
            )
            projected_inflows = round(max(firm_inflows, expected_run_rate_inflows), 2)

            # 5. Projected Outflows: Baseline Burn + Scheduled Maintenance
            baseline_monthly_burn = baseline_monthly_inflow * 0.35
            expected_run_rate_burn = baseline_monthly_burn * month_factor
            projected_outflows = round(
                max(non_maintenance_outflows, expected_run_rate_burn)
                + scheduled_maintenance_outflow,
                2,
            )

            # 6. Balance & Solvency Audit
            current_cash_balance = profile["starting_cash"]
            minimum_threshold = profile["safety_floor"]

            projected_end_balance = round(
                current_cash_balance + projected_inflows - projected_outflows, 2
            )
            shortfall_risk = projected_end_balance < minimum_threshold

            return {
                "account_id": acc_key,
                "current_cash_balance": current_cash_balance,
                "projected_end_balance": projected_end_balance,
                "projected_inflows": projected_inflows,
                "projected_outflows": projected_outflows,
                # Legacy keys kept for UI backward compatibility
                "projected_end_balance_30d": projected_end_balance,
                "projected_inflows_30d": projected_inflows,
                "projected_outflows_30d": projected_outflows,
                "minimum_threshold": minimum_threshold,
                "shortfall_risk_detected": shortfall_risk,
                "ml_engine_active": (
                    model_bundle is not None
                    and model_bundle.get("approved", False)
                ),
                "model_version": "v2_residual_lightgbm",
                "predicted_realization_ratio": round(predicted_ratio, 4),
                "confidence_band_80": {
                    "downside_usd": round(projected_inflows * (1.0 + band_low), 2),
                    "upside_usd": round(projected_inflows * (1.0 + band_high), 2),
                    "observed_validation_coverage": (
                        round(float(model_bundle.get("residual_band_observed_coverage", 0.0)), 4)
                        if model_bundle else None
                    ),
                },
                "horizon_days": days,
                "target_date": target_horizon.strftime("%b %d, %Y"),
            }
        finally:
            if should_close:
                session.close()

    # Horizon-specific helper wrappers
    def calculate_30d_summary(
        self,
        db: Optional[Session] = None,
        days: int = 30,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=days, account_id=account_id, db=db)

    def calculate_60d_summary(
        self,
        db: Optional[Session] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=60, account_id=account_id, db=db)

    def calculate_90d_summary(
        self,
        db: Optional[Session] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=90, account_id=account_id, db=db)

    def get_summary_30d(
        self,
        db: Optional[Session] = None,
        days: int = 30,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=days, account_id=account_id, db=db)

    def get_summary_60d(
        self,
        db: Optional[Session] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=60, account_id=account_id, db=db)

    def get_summary_90d(
        self,
        db: Optional[Session] = None,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=90, account_id=account_id, db=db)

    def get_summary(
        self,
        db: Optional[Session] = None,
        days: int = 30,
        account_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.calculate_forecast(days=days, account_id=account_id, db=db)


CashFlowForecaster = LiquidityForecaster
