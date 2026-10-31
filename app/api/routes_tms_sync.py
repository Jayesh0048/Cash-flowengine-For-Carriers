from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.engine.tms_auth_client import TMSUserClient

router = APIRouter(prefix="/sync", tags=["TMS Live Synchronization"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/user-tms")
def trigger_user_tms_sync(db: Session = Depends(get_db)):
    """Triggers authentication for santosh@bigwings.us and pulls live loads/expenses."""
    client = TMSUserClient()
    result = client.sync_to_database(db=db)
    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Failed to sync TMS records for user."),
        )
    return result


@router.get("/status")
def get_sync_status():
    """Returns current active synchronization credentials configuration."""
    client = TMSUserClient()
    return {
        "sync_user": client.email,
        "target_account": client.account_id,
        "endpoint_configured": bool(client.base_url),
    }