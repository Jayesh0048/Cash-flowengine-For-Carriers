import os
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.engine.ingestion import TMSIngestionService

load_dotenv()


class TMSUserClient:
    """Authenticates as a dedicated user and pulls live loads/expenses from the TMS."""

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        login_url: Optional[str] = None,
        base_url: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        self.email = email or os.getenv("TMS_USER_EMAIL", "")
        self.password = password or os.getenv("TMS_USER_PASSWORD", "")
        self.login_url = login_url or os.getenv(
            "TMS_LOGIN_URL", "https://api.etruckingsystem.com/api/v1/auth/login"
        )
        self.base_url = base_url or os.getenv(
            "TMS_BASE_URL", "https://api.etruckingsystem.com/api/v1"
        )
        self.account_id = account_id or os.getenv("TMS_DEFAULT_ACCOUNT", "acc1")

        self.session = requests.Session()
        self.token: Optional[str] = None

    def authenticate(self) -> bool:
        """Submits user credentials and extracts the session JWT/Bearer token."""
        if not self.email or not self.password:
            print("[!] TMS_USER_EMAIL and TMS_USER_PASSWORD must be configured.")
            return False
        login_payload = {
            "email": self.email,
            "username": self.email,
            "password": self.password,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            resp = self.session.post(
                self.login_url, json=login_payload, headers=headers, timeout=12
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                # Dynamically locate the authentication token
                self.token = (
                    data.get("access_token")
                    or data.get("token")
                    or data.get("jwt")
                    or data.get("id_token")
                    or (data.get("data", {}).get("token") if isinstance(data.get("data"), dict) else None)
                )

                if self.token:
                    self.session.headers.update({"Authorization": f"Bearer {self.token}"})

                print(f"[✓] Successfully authenticated as: {self.email}")
                return True
            else:
                print(f"[!] Authentication failed ({resp.status_code}): {resp.text}")
                return False

        except Exception as err:
            print(f"[!] Network or URL error during authentication: {err}")
            return False

    def fetch_user_loads(self, limit: int = 250) -> List[Dict[str, Any]]:
        """Pulls loads associated with this authenticated account."""
        if not self.token:
            if not self.authenticate():
                return []

        url = f"{self.base_url.rstrip('/')}/loads"
        params = {"limit": limit}

        try:
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                body = resp.json()
                loads = body.get("data") or body.get("loads") or body
                if isinstance(loads, list):
                    print(f"[✓] Fetched {len(loads)} loads for {self.email}")
                    return loads
                return []
            elif resp.status_code == 401:
                # Token expired, retry once
                print("[!] Token expired. Re-authenticating...")
                if self.authenticate():
                    retry_resp = self.session.get(url, params=params, timeout=15)
                    if retry_resp.status_code == 200:
                        body = retry_resp.json()
                        return body.get("data") or body.get("loads") or []
            print(f"[!] Failed to fetch loads ({resp.status_code}): {resp.text}")
            return []
        except Exception as err:
            print(f"[!] Error fetching loads: {err}")
            return []

    def fetch_user_expenses(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Pulls operational expenses associated with this authenticated account."""
        if not self.token:
            if not self.authenticate():
                return []

        url = f"{self.base_url.rstrip('/')}/expenses"
        params = {"limit": limit}

        try:
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                body = resp.json()
                expenses = body.get("data") or body.get("expenses") or body
                if isinstance(expenses, list):
                    print(f"[✓] Fetched {len(expenses)} expenses for {self.email}")
                    return expenses
                return []
            return []
        except Exception:
            return []

    def sync_to_database(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Orchestrates authentication, retrieval, and database ingestion."""
        session = db if db is not None else SessionLocal()
        should_close = db is None

        try:
            if not self.authenticate():
                return {
                    "success": False,
                    "error": f"Failed to authenticate as {self.email}",
                    "loads_synced": 0,
                    "expenses_synced": 0,
                }

            loads = self.fetch_user_loads()
            expenses = self.fetch_user_expenses()

            service = TMSIngestionService()
            load_result = {"new_events_recorded": 0, "skipped_existing": 0}
            expense_result = {"new_events_recorded": 0, "skipped_existing": 0}

            if loads:
                load_result = service.process_and_store_loads(
                    loads, session, prefix=self.account_id
                )
            if expenses:
                expense_result = service.process_and_store_expenses(
                    expenses, session, prefix=self.account_id
                )

            return {
                "success": True,
                "authenticated_user": self.email,
                "account_id": self.account_id,
                "loads_fetched": len(loads),
                "loads_new_recorded": load_result.get("new_events_recorded", 0),
                "expenses_fetched": len(expenses),
                "expenses_new_recorded": expense_result.get("new_events_recorded", 0),
                "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
        finally:
            if should_close:
                session.close()


if __name__ == "__main__":
    client = TMSUserClient()
    result = client.sync_to_database()
    print("[*] Sync Result:\n", json.dumps(result, indent=2))
