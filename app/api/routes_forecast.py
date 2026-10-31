from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.core.database import SessionLocal
from app.core.models import CashEvent
from app.engine.forecaster import LiquidityForecaster
from app.engine.ingestion import TMSIngestionService

router = APIRouter(prefix="/forecast", tags=["Forecast"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


forecaster = LiquidityForecaster()


@router.get("/summary")
def get_forecast_summary(
    days: int = Query(30, ge=1, le=365, description="Forecast horizon in days"),
    account_id: str = Query("all", pattern="^(all|acc1|acc2)$", description="Fleet account"),
    db: Session = Depends(get_db)
):
    return forecaster.calculate_30d_summary(db=db, days=days, account_id=account_id)


@router.get("/events")
def get_cash_events(
    limit: int = Query(500, description="Maximum number of ledger records to return"),
    db: Session = Depends(get_db)
):
    # Sort with newest and upcoming events first
    events = (
        db.query(CashEvent)
        .order_by(CashEvent.due_date.desc())
        .limit(limit)
        .all()
    )
    return events


@router.post("/refresh-all")
def refresh_all_pipeline(db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = TMSIngestionService()
    loads = service.fetch_loads_from_csv("master_loads.csv")
    res_loads = service.process_and_store_loads(loads, db) if loads else {}

    expenses = service.fetch_expenses_from_csv("master_expenses.csv")
    res_expenses = service.process_and_store_expenses(expenses, db) if expenses else {}

    return {
        "status": "success",
        "loads": res_loads,
        "expenses": res_expenses
    }


@router.post("/sync-loads")
def sync_loads_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = TMSIngestionService()
    loads = service.fetch_loads_from_csv("master_loads.csv")
    res = service.process_and_store_loads(loads, db) if loads else {}
    return {"status": "success", "result": res}


@router.post("/sync-expenses")
def sync_expenses_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    service = TMSIngestionService()
    expenses = service.fetch_expenses_from_csv("master_expenses.csv")
    res = service.process_and_store_expenses(expenses, db) if expenses else {}
    return {"status": "success", "result": res}
