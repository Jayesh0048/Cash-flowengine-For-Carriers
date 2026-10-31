"""Download expense records from the TMS using a session cookie from the environment."""

import os
import time

import pandas as pd
import requests

API_URL = "https://etruckingsystem.com/tms/accounting/getExpenses"


def extract_api_data() -> None:
    auth_cookie = os.getenv("TMS_AUTH_COOKIE", "")
    if not auth_cookie:
        print("[!] Set TMS_AUTH_COOKIE before running this extractor.")
        return

    headers = {
        "Cookie": auth_cookie,
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "CashFlowEngine/1.0",
    }
    records = []
    offset = 1

    while True:
        try:
            response = requests.get(
                API_URL, headers=headers, params={"limit": 20, "offset": offset}, timeout=15
            )
            response.raise_for_status()
            body = response.json()
            items = body.get("data") or body.get("items") or body.get("expenses") or []
            if not items:
                break
            records.extend(items)
            offset += 1
            time.sleep(0.5)
        except requests.RequestException as error:
            print(f"[!] TMS request failed: {error}")
            return

    if records:
        pd.DataFrame(records).to_csv("expenses.csv", index=False)
        print(f"[✓] Saved {len(records)} expense records to expenses.csv")
    else:
        print("[*] No expense records returned.")


if __name__ == "__main__":
    extract_api_data()
