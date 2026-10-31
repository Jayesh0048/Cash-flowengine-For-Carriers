from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class TMSLoadItem(BaseModel):
    load_id: str = Field(..., description="Unique load identifier from TMS")
    load_number: str
    status: str
    rate: float
    carrier_pay: float = 0.0
    broker_name: Optional[str] = "Unknown"
    pickup_date: Optional[datetime] = None
    delivery_date: Optional[datetime] = None
    payment_terms_days: int = 30
    is_factored: bool = True
    factoring_fee_pct: float = 0.03
    reserve_escrow_pct: float = 0.05

    model_config = ConfigDict(from_attributes=True)


class DriverPayItem(BaseModel):
    driver_id: str
    driver_name: str
    amount: float
    pay_date: datetime

    model_config = ConfigDict(from_attributes=True)


class ExpenseItem(BaseModel):
    expense_id: str
    label: str
    amount: float
    due_date: datetime

    model_config = ConfigDict(from_attributes=True)


class MaintenanceWorkOrderItem(BaseModel):
    work_order_id: str
    asset_id: str
    amount: float
    target_date: datetime
    status: str = "approved"

    model_config = ConfigDict(from_attributes=True)


class SyncResponse(BaseModel):
    total_fetched: int
    new_events_recorded: int
    skipped_existing: int
    sync_timestamp: datetime


TMSLoadSyncResponse = SyncResponse


class CashEventResponse(BaseModel):
    id: int
    event_id: str
    account_id: Optional[str] = "acc1"
    source_type: str
    direction: str
    category: str
    gross_amount: float
    net_amount: float
    fee_deductions: float
    status: str
    due_date: datetime

    model_config = ConfigDict(from_attributes=True)


class ForecastSummaryResponse(BaseModel):
    current_cash_balance: float
    projected_inflows_30d: float
    projected_outflows_30d: float
    projected_end_balance_30d: float
    minimum_threshold: float
    shortfall_risk_detected: bool
    ml_engine_active: Optional[bool] = True
    model_version: Optional[str] = "v2_residual_lightgbm"
    horizon_days: Optional[int] = 30
    target_date: Optional[str] = None
    predicted_realization_ratio: Optional[float] = None
    confidence_band_80: Optional[Dict[str, float]] = None

    model_config = ConfigDict(from_attributes=True)