from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core import models  # noqa: F401
from app.core.database import Base
from app.core.models import CashEvent
from app.engine.forecaster import LiquidityForecaster


def test_forecast_uses_only_requested_account():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    due = now + timedelta(days=2)
    session.add_all([
        CashEvent(event_id="acc1-settled", account_id="acc1", source_type="LOAD", direction="INFLOW", gross_amount=1000, net_amount=920, due_date=now - timedelta(days=10), status="SETTLED"),
        CashEvent(event_id="acc1-load", account_id="acc1", source_type="LOAD", direction="INFLOW", gross_amount=1000, net_amount=920, due_date=due),
        CashEvent(event_id="acc2-settled", account_id="acc2", source_type="LOAD", direction="INFLOW", gross_amount=9000, net_amount=8280, due_date=now - timedelta(days=10), status="SETTLED"),
        CashEvent(event_id="acc2-load", account_id="acc2", source_type="LOAD", direction="INFLOW", gross_amount=9000, net_amount=8280, due_date=due),
    ])
    session.commit()

    forecaster = LiquidityForecaster()
    forecaster.model_bundle = None
    result = forecaster.calculate_forecast(days=30, account_id="acc1", db=session)

    assert result["account_id"] == "acc1"
    assert result["projected_inflows"] < 2_000
    assert result["horizon_days"] == 30
