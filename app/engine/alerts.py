"""
Sends a shortfall warning when the forecast summary shows the projected
balance dropping below the minimum reserve threshold. If TEAMS_WEBHOOK_URL
isn't set yet, it just logs instead of failing.
"""
import httpx
from app.config import settings


def send_shortfall_alert(summary: dict) -> None:
    if not summary.get("shortfall_risk_detected"):
        return

    message = (
        f"Cash flow warning: projected 30-day end balance is "
        f"${summary['projected_end_balance_30d']:,.2f}, below the "
        f"${summary['minimum_threshold']:,.2f} minimum reserve. "
        f"Inflows: ${summary['projected_inflows_30d']:,.2f}, "
        f"Outflows: ${summary['projected_outflows_30d']:,.2f}."
    )

    if not settings.TEAMS_WEBHOOK_URL:
        print(f"[ALERT - no Teams webhook configured] {message}")
        return

    try:
        httpx.post(settings.TEAMS_WEBHOOK_URL, json={"text": message}, timeout=10.0)
    except Exception as err:
        print(f"[ALERT FAILED] Could not reach Teams webhook: {err}")