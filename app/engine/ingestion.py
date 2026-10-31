import os
import json
import math
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
from app.core.models import CashEvent

logger = logging.getLogger("ingestion_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_clean_utc_date(val: Optional[Any], default_now: datetime) -> datetime:
    """Robustly parse any string/timestamp date format into a UTC-aware datetime."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default_now
    try:
        ts = pd.to_datetime(val, utc=True)
        if pd.isna(ts):
            return default_now
        return ts.to_pydatetime()
    except Exception:
        return default_now


class TMSIngestionService:
    def __init__(self, batch_chunk_size: int = 500):
        self.chunk_size = batch_chunk_size

    def fetch_loads_from_csv(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            logger.warning(f"CSV file not found: {file_path}")
            return []
        try:
            df = pd.read_csv(file_path)
            df = df.where(pd.notnull(df), None)
            records = df.to_dict(orient="records")
            logger.info(f"Loaded {len(records)} loads from {file_path}")
            return records
        except Exception as err:
            logger.error(f"Error reading {file_path}: {err}", exc_info=True)
            return []

    def fetch_expenses_from_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Read an expense CSV, tolerating common encodings and blank values."""
        if not os.path.exists(file_path):
            logger.warning("Expense CSV file not found: %s", file_path)
            return []
        if os.path.getsize(file_path) == 0:
            logger.warning("Expense CSV file is empty: %s", file_path)
            return []

        df = None
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                df = pd.read_csv(file_path, encoding=encoding, dtype=str)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.EmptyDataError:
                logger.warning("Expense CSV has no data: %s", file_path)
                return []
            except Exception as err:
                logger.error("Error parsing %s: %s", file_path, err)
                return []

        if df is None or df.empty:
            logger.info("Zero expense records found in %s", file_path)
            return []

        df.columns = [str(column).strip() for column in df.columns]
        df = df.map(lambda value: value.strip() if isinstance(value, str) else value)
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        logger.info("Loaded %s expense records from %s", len(records), file_path)
        return records

    def process_and_store_loads(self, loads_raw: List[Dict], db: Session, prefix: str = "acc1") -> Dict[str, int]:
        total_fetched = len(loads_raw)
        new_recorded = 0
        skipped = 0
        now_utc = datetime.now(timezone.utc)

        # 1. Pre-fetch existing IDs into memory to avoid thousands of DB round-trips
        existing_ids = set(
            row[0] for row in db.query(CashEvent.event_id).filter(CashEvent.account_id == prefix).all()
        )
        seen_in_batch = set()
        escrow_batches = defaultdict(lambda: {"amount": 0.0, "count": 0, "due_date": None})

        try:
            for index, raw_item in enumerate(loads_raw):
                raw_id = (
                    raw_item.get("loadId") or 
                    raw_item.get("id") or 
                    raw_item.get("loadNumber") or 
                    raw_item.get("workOrder") or
                    f"idx_{index}"
                )
                load_id = f"{prefix}_{raw_id}"

                raw_rate = (
                    raw_item.get("rateReceived") or 
                    raw_item.get("totalAmount") or 
                    raw_item.get("rate") or 
                    0.0
                )
                try:
                    cleaned_str = str(raw_rate).replace("$", "").replace(",", "").strip()
                    rate = float(cleaned_str) if cleaned_str and not math.isnan(float(cleaned_str)) else 0.0
                except (ValueError, TypeError):
                    rate = 0.0

                if rate <= 0.0:
                    skipped += 1
                    continue

                broker_name = str(
                    raw_item.get("customerType") or 
                    raw_item.get("Customer") or 
                    raw_item.get("broker") or 
                    raw_item.get("debtorName") or
                    "Broker"
                )

                delivery_str = (
                    raw_item.get("loadEndTime") or 
                    raw_item.get("endTime") or 
                    raw_item.get("createdAt") or 
                    raw_item.get("deliveryDate") or
                    raw_item.get("startTime")
                )
                delivery_date = parse_clean_utc_date(delivery_str, default_now=now_utc)

                event_id = f"load_{load_id}"
                if event_id in existing_ids or event_id in seen_in_batch:
                    skipped += 1
                    continue

                seen_in_batch.add(event_id)
                existing_ids.add(event_id)

                # Factoring Terms: 3% fee, 5% escrow hold
                factoring_fee_pct = 0.03
                reserve_escrow_pct = 0.05

                fee = round(rate * factoring_fee_pct, 2)
                escrow = round(rate * reserve_escrow_pct, 2)
                immediate_net = round(rate - fee - escrow, 2)

                payout_due = delivery_date + timedelta(days=2)
                escrow_due = payout_due + timedelta(days=30)
                is_settled = payout_due < now_utc

                db.add(CashEvent(
                    event_id=event_id,
                    account_id=prefix,
                    source_type="LOAD",
                    direction="INFLOW",
                    category=f"Freight Revenue - {broker_name}",
                    gross_amount=rate,
                    net_amount=immediate_net,
                    fee_deductions=fee + escrow,
                    status="SETTLED" if is_settled else "PENDING",
                    due_date=payout_due,
                    raw_payload=json.dumps(raw_item, default=str),
                ))
                new_recorded += 1

                # Clean UTC normalization without string round-tripping
                if escrow > 0:
                    day_key = escrow_due.strftime("%Y-%m-%d")
                    normalized_due = escrow_due.replace(hour=0, minute=0, second=0, microsecond=0)
                    escrow_batches[day_key]["amount"] += escrow
                    escrow_batches[day_key]["count"] += 1
                    escrow_batches[day_key]["due_date"] = normalized_due

            # 2. Store or update escrow batches
            for day_key, batch_data in escrow_batches.items():
                batch_amount = round(batch_data["amount"], 2)
                if batch_amount <= 0:
                    continue

                batch_event_id = f"escrow_batch_{prefix}_{day_key}"
                existing_batch = db.query(CashEvent).filter(
                    CashEvent.event_id == batch_event_id,
                    CashEvent.account_id == prefix
                ).first()

                if existing_batch:
                    existing_batch.gross_amount = round(existing_batch.gross_amount + batch_amount, 2)
                    existing_batch.net_amount = round(existing_batch.net_amount + batch_amount, 2)
                    existing_batch.category = f"Factoring Reserve Release Batch ({day_key})"
                else:
                    is_batch_settled = batch_data["due_date"] < now_utc
                    db.add(CashEvent(
                        event_id=batch_event_id,
                        account_id=prefix,
                        source_type="FACTORING",
                        direction="INFLOW",
                        category=f"Factoring Reserve Release Batch ({batch_data['count']} loads)",
                        gross_amount=batch_amount,
                        net_amount=batch_amount,
                        fee_deductions=0.0,
                        status="SETTLED" if is_batch_settled else "PENDING",
                        due_date=batch_data["due_date"],
                        raw_payload=json.dumps({"consolidated_loads": batch_data["count"]}, default=str),
                    ))
                    new_recorded += 1

            db.commit()
            return {"total_fetched": total_fetched, "new_events_recorded": new_recorded, "skipped_existing": skipped}

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process loads for {prefix}: {e}", exc_info=True)
            raise

    def process_and_store_expenses(self, items_raw: List[Dict], db: Session, prefix: str = "acc1") -> Dict[str, int]:
        total_fetched = len(items_raw)
        new_recorded = 0
        skipped = 0
        now_utc = datetime.now(timezone.utc)

        existing_ids = set(
            row[0] for row in db.query(CashEvent.event_id).filter(CashEvent.account_id == prefix).all()
        )
        seen_in_batch = set()

        try:
            for index, raw_item in enumerate(items_raw):
                raw_id = (
                    raw_item.get("expenseId") or 
                    raw_item.get("id") or 
                    raw_item.get("referenceNumber") or
                    f"exp_{index}"
                )
                expense_id = f"{prefix}_{raw_id}"

                raw_amt = (
                    raw_item.get("totalCost") or 
                    raw_item.get("totalAmount") or 
                    raw_item.get("amount") or 
                    0.0
                )
                try:
                    cleaned_amt = str(raw_amt).replace("$", "").replace(",", "").strip()
                    amount = abs(float(cleaned_amt)) if cleaned_amt and not math.isnan(float(cleaned_amt)) else 0.0
                except (ValueError, TypeError):
                    amount = 0.0

                if amount <= 0.0:
                    skipped += 1
                    continue

                label = str(
                    raw_item.get("documentType") or 
                    raw_item.get("description") or 
                    raw_item.get("category") or
                    "Operational Expense"
                )

                date_str = raw_item.get("date") or raw_item.get("createdAt") or raw_item.get("expenseDate")
                due_date = parse_clean_utc_date(date_str, default_now=now_utc)

                event_id = f"expense_{expense_id}"
                if event_id in existing_ids or event_id in seen_in_batch:
                    skipped += 1
                    continue

                seen_in_batch.add(event_id)
                existing_ids.add(event_id)

                is_settled = due_date < now_utc

                db.add(CashEvent(
                    event_id=event_id,
                    account_id=prefix,
                    source_type="EXPENSE",
                    direction="OUTFLOW",
                    category=f"Expense - {label}",
                    gross_amount=amount,
                    net_amount=amount,
                    fee_deductions=0.0,
                    status="SETTLED" if is_settled else "PENDING",
                    due_date=due_date,
                    raw_payload=json.dumps(raw_item, default=str),
                ))
                new_recorded += 1

            db.commit()
            return {"total_fetched": total_fetched, "new_events_recorded": new_recorded, "skipped_existing": skipped}

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process expenses for {prefix}: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Safe non-destructive initialization: creates tables only if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    service = TMSIngestionService()

    try:
        # Account 1 Ingestion
        for file_name, method in [("master_loads.csv", service.process_and_store_loads), 
                                 ("master_expenses.csv", service.process_and_store_expenses)]:
            if os.path.exists(file_name):
                records = (service.fetch_loads_from_csv if "load" in file_name else service.fetch_expenses_from_csv)(file_name)
                res = method(records, db, prefix="acc1")
                logger.info(f"Account 1 -> {file_name}: {res}")

        # Account 2 Ingestion
        for file_name, method in [("account2_all_loads.csv", service.process_and_store_loads),
                                 ("account2_all_loads (1).csv", service.process_and_store_loads),
                                 ("account2_expenses.csv", service.process_and_store_expenses)]:
            if os.path.exists(file_name):
                records = (service.fetch_loads_from_csv if "load" in file_name else service.fetch_expenses_from_csv)(file_name)
                res = method(records, db, prefix="acc2")
                logger.info(f"Account 2 -> {file_name}: {res}")

    finally:
        db.close()
        logger.info("Multi-fleet ingestion completed successfully.")
