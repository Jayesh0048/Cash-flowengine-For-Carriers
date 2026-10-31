import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

# Core API Routers
from app.api.routes_forecast import router as forecast_router
from app.api.routes_tms_sync import router as sync_router
from app.api.routes_tms_webhook import router as webhook_router
from app.core.database import Base, engine
from app.core import models  # noqa: F401 -- registers SQLAlchemy models before table creation

# Optional: include simulation/agent routes if configured
try:
    from app.api.routes_agents import router as agents_router
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False

try:
    from app.api.routes_analytics import router as analytics_router
    HAS_ANALYTICS = True
except ImportError:
    HAS_ANALYTICS = False


app = FastAPI(
    title="TMS Liquidity & Compliance Engine API",
    version="2.0.0",
    description="Headless intelligence microservice providing cash flow forecasting, factoring reconciliation, and compliance simulation.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for TMS web client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your TMS domain (e.g., ["https://tms.bigwings.us"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The local SQLite database is usable on a fresh checkout without a manual
# migration step.  Production deployments should replace this with migrations.
Base.metadata.create_all(bind=engine)

# 1. Mount Cash Flow & Forecasting Endpoints
app.include_router(forecast_router, prefix="/api/v1")

# 2. Mount Live TMS User Synchronization (santosh@bigwings.us)
app.include_router(sync_router, prefix="/api/v1")

# 3. Mount Real-Time TMS Webhook Listener
app.include_router(webhook_router, prefix="/api/v1")

# 4. Mount Supplementary Agents (Maintenance, Safety, Analytics)
if HAS_AGENTS:
    app.include_router(agents_router, prefix="/api/v1")

if HAS_ANALYTICS:
    app.include_router(analytics_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects base traffic directly to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Serve the bundled read-only cash-flow dashboard."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))


@app.get("/health", tags=["System Health"])
def health_check():
    """Health check endpoint for TMS microservice orchestrators."""
    return {
        "status": "online",
        "engine": "Residual LightGBM v2",
        "authenticated_sync_account": "santosh@bigwings.us",
        "mode": "headless_api",
    }
