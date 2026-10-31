from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from app.core.database import Base


class CashEvent(Base):
    __tablename__ = "cash_events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    account_id = Column(String(32), default="acc1", index=True, nullable=True)  # <-- Added missing column
    source_type = Column(String(32), index=True)  # LOAD, FACTORING, DRIVER PAY, FUEL, MAINTENANCE
    direction = Column(String(16))  # INFLOW, OUTFLOW
    category = Column(String(64))
    gross_amount = Column(Float, default=0.0)
    net_amount = Column(Float, default=0.0)
    fee_deductions = Column(Float, default=0.0)
    status = Column(String(32), default="PENDING")  # PENDING, SETTLED, CANCELLED
    due_date = Column(DateTime, index=True)
    settled_date = Column(DateTime, nullable=True)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))