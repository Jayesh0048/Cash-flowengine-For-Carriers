import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.core.models import CashEvent

router = APIRouter(prefix="/webhooks", tags=["TMS Live Webhooks"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TMSLoadPayload(BaseModel):
    event_type: str  # e.g., "load.created", "load.delivered", "load.billed"
    account_id: str = "acc1"  # "acc1" or "acc2"
    load_id: str
    rate: float
    broker_name: str
    delivery_date: Optional[str] = None  # ISO format string


@router.post("/tms-event")
async def receive_tms_event(
    payload: TMSLoadPayload,
    db: Session = Depends(get_db),
    x_tms_api_key: Optional[str] = Header(None),
):
    """Webhook listener that captures real-time freight load events

    and automatically injects them into the Cash Flow Engine.
    """
    # 1. Parse Delivery Timestamp
    now_utc = datetime.now(timezone.utc)
    if payload.delivery_date:
        try:
            delivery_dt = datetime.fromisoformat(payload.delivery_date.replace("Z", "+00:00"))
        except Exception:
            delivery_dt = now_utc
    else:
        delivery_dt = now_utc

    delivery_naive = delivery_dt.replace(tzinfo=None)

    # Factoring Terms: 3% fee, 5% escrow hold, 92% upfront advance
    rate = float(payload.rate or 0.0)
    factoring_fee = round(rate * 0.03, 2)
    escrow_reserve = round(rate * 0.05, 2)
    immediate_advance = round(rate - factoring_fee - escrow_reserve, 2)

    payout_due = delivery_naive + timedelta(days=2)
    escrow_due = payout_due + timedelta(days=30)
    now_naive = now_utc.replace(tzinfo=None)

    # 2. Inflow 1: 92% Upfront Factoring Advance
    event_id = f"tms_load_{payload.account_id}_{payload.load_id}"
    existing_load = db.query(CashEvent).filter(CashEvent.event_id == event_id).first()

    if existing_load:
        # Update existing record if rate changed in TMS
        existing_load.gross_amount = rate
        existing_load.net_amount = immediate_advance
        existing_load.fee_deductions = factoring_fee + escrow_reserve
        existing_load.due_date = payout_due
        existing_load.status = "SETTLED" if payout_due < now_naive else "PENDING"
    else:
        db.add(CashEvent(
            event_id=event_id,
            account_id=payload.account_id,
            source_type="LOAD",
            direction="INFLOW",
            category=f"Freight Revenue - {payload.broker_name}",
            gross_amount=rate,
            net_amount=immediate_advance,
            fee_deductions=factoring_fee + escrow_reserve,
            status="SETTLED" if payout_due < now_naive else "PENDING",
            due_date=payout_due,
            raw_payload=json.dumps(payload.dict(), default=str),
        ))

    # 3. Inflow 2: Aggregate into Daily Escrow Batch
    if escrow_reserve > 0:
        day_key = escrow_due.strftime("%Y-%m-%d")
        batch_event_id = f"escrow_batch_{payload.account_id}_{day_key}"
        existing_batch = db.query(CashEvent).filter(CashEvent.event_id == batch_event_id).first()

        if existing_batch:
            existing_batch.gross_amount = round(existing_batch.gross_amount + escrow_reserve, 2)
            existing_batch.net_amount = round(existing_batch.net_amount + escrow_reserve, 2)
        else:
            db.add(CashEvent(
                event_id=batch_event_id,
                account_id=payload.account_id,
                source_type="FACTORING",
                direction="INFLOW",
                category=f"Factoring Reserve Release Batch ({day_key})",
                gross_amount=escrow_reserve,
                net_amount=escrow_reserve,
                fee_deductions=0.0,
                status="SETTLED" if escrow_due < now_naive else "PENDING",
                due_date=escrow_due,
                raw_payload=json.dumps({"origin": "live_tms_webhook"}),
            ))

    db.commit()

    return {
        "status": "PROCESSED",
        "load_id": payload.load_id,
        "account_id": payload.account_id,
        "advance_payout": immediate_advance,
        "escrow_hold": escrow_reserve,
        "payout_due_date": payout_due.strftime("%Y-%m-%d"),
    }