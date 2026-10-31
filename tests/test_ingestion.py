from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core import models  # noqa: F401
from app.core.database import Base
from app.core.models import CashEvent
from app.engine.ingestion import TMSIngestionService


def test_service_exposes_expense_loader_and_processors(tmp_path):
    expense_file = tmp_path / "expenses.csv"
    expense_file.write_text("expenseId,totalCost,date,description\nE-1,$125.50,2026-10-01,Fuel\n")

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    service = TMSIngestionService()

    records = service.fetch_expenses_from_csv(str(expense_file))
    result = service.process_and_store_expenses(records, session, prefix="acc1")

    assert result["new_events_recorded"] == 1
    event = session.query(CashEvent).one()
    assert event.event_id == "expense_acc1_E-1"
    assert event.direction == "OUTFLOW"
    assert event.net_amount == 125.50
