import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.core.models import (
    CashEvent,
    DiagnosticCode,
    DVIRLog,
    ServiceCenter,
    VehicleAsset,
)


class PredictiveMaintenanceAgent:
    """Autonomous fleet health diagnostic and predictive maintenance engine.

    Fuses unstructured driver inspection logs (NLP), J1939 telematics diagnostic trouble codes (DTCs),
    and odometer wear metrics to forecast component failure risks, route vehicles to preferred service
    hubs, and automatically inject predictive repair expenses into the Cash Flow Engine.
    """

    # Industry baseline costs and repair windows by subsystem
    SUBSYSTEM_PROFILES = {
        "BRAKES": {
            "default_repair_cost": 1250.0,
            "turnaround_days": 2,
            "criticality_weight": 0.40,
        },
        "COOLING": {
            "default_repair_cost": 850.0,
            "turnaround_days": 3,
            "criticality_weight": 0.30,
        },
        "AIR_SYSTEM": {
            "default_repair_cost": 950.0,
            "turnaround_days": 2,
            "criticality_weight": 0.35,
        },
        "TIRES": {
            "default_repair_cost": 650.0,
            "turnaround_days": 1,
            "criticality_weight": 0.25,
        },
        "AFTERTREATMENT": {
            "default_repair_cost": 3200.0,
            "turnaround_days": 4,
            "criticality_weight": 0.45,
        },
        "GENERAL": {
            "default_repair_cost": 750.0,
            "turnaround_days": 2,
            "criticality_weight": 0.20,
        },
    }

    def __init__(self, db: Session):
        self.db = db

        # Truck subsystem symptom keywords
        self.symptom_lexicon = {
            "BRAKES": [
                "soft brake", "air leak", "slack adjuster", "squeal",
                "brake smell", "pulls left", "lining", "pad worn", "grinding",
            ],
            "COOLING": [
                "coolant", "temp high", "overheating", "radiator leak",
                "sweet smell", "water pump", "hose leak", "antifreeze",
            ],
            "AIR_SYSTEM": [
                "psi drop", "governor", "slow build", "air compressor",
                "purge valve", "low air pressure", "air tank",
            ],
            "TIRES": [
                "tread low", "vibration", "sidewall", "valve stem",
                "low psi", "tire heat", "blowout", "flat tire", "tread separation",
            ],
            "AFTERTREATMENT": [
                "def light", "regen", "dpf", "nox", "scr derate",
                "exhaust smell", "soot", "emissions warning",
            ],
        }

        # Negation triggers to prevent false positives (e.g., "no brake squeal")
        self.negation_patterns = [
            r"\bno\s+",
            r"\bnot\s+",
            r"\bwithout\s+",
            r"\bchecked\s+no\s+",
            r"\bpassed\s+",
        ]

    # ============================================================
    # 1. NLP DVIR DEFECT CLASSIFICATION
    # ============================================================

    def parse_dvir_nlp(self, notes: str) -> Dict[str, Any]:
        """Classifies driver inspection text, removes negated phrases,

        and scores failure urgency.
        """
        if not notes or not notes.strip():
            return {
                "subsystem": "NONE",
                "severity": 0.0,
                "urgency": "NORMAL",
                "has_defect": False,
                "detected_symptoms": [],
            }

        clean_text = notes.lower().strip()
        detected_symptoms: List[str] = []
        scores_by_system: Dict[str, int] = {k: 0 for k in self.symptom_lexicon}

        # Break notes into sentence clauses to handle negations accurately
        clauses = re.split(r"[,.;\n]+", clean_text)

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            for system, keywords in self.symptom_lexicon.items():
                for kw in keywords:
                    if re.search(r"\b" + re.escape(kw) + r"\b", clause):
                        # Verify the phrase isn't negated in this clause
                        is_negated = any(
                            re.search(pat + re.escape(kw), clause)
                            for pat in self.negation_patterns
                        )
                        if not is_negated:
                            scores_by_system[system] += 1
                            detected_symptoms.append(f"{system}: {kw}")

        # Determine primary damaged subsystem
        best_system = "GENERAL"
        max_matches = 0
        for sys_name, match_count in scores_by_system.items():
            if match_count > max_matches:
                max_matches = match_count
                best_system = sys_name

        if max_matches == 0:
            return {
                "subsystem": "NONE",
                "severity": 0.0,
                "urgency": "NORMAL",
                "has_defect": False,
                "detected_symptoms": [],
            }

        # High-threat emergency indicators
        critical_indicators = [
            "danger", "derate", "smoke", "grinding", "shutting down",
            "blown", "flashing", "fail", "fire", "locked up",
        ]
        emergency_hits = sum(
            1 for word in critical_indicators
            if re.search(r"\b" + re.escape(word) + r"\b", clean_text)
        )

        base_severity = min(0.60, max_matches * 0.20)
        critical_severity = min(0.40, emergency_hits * 0.20)
        final_severity = round(min(1.0, base_severity + critical_severity), 2)

        if final_severity >= 0.70:
            urgency = "CRITICAL"
        elif final_severity >= 0.35:
            urgency = "MODERATE"
        else:
            urgency = "MINOR"

        return {
            "subsystem": best_system,
            "severity": final_severity,
            "urgency": urgency,
            "has_defect": True,
            "detected_symptoms": detected_symptoms,
        }

    # ============================================================
    # 2. VEHICLE COMPOSITE HEALTH & RISK SCORING
    # ============================================================

    def assess_vehicle_health(self, unit_number: str) -> Dict[str, Any]:
        """Calculates a composite health score (0-100) using:

        - Active DTC fault codes (last 14 days)
        - Driver DVIR inspection history (last 14 days)
        - Odometer wear index
        """
        asset = (
            self.db.query(VehicleAsset)
            .filter(VehicleAsset.unit_number == unit_number)
            .first()
        )
        if not asset:
            return {"error": f"Vehicle {unit_number} not found"}

        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        two_weeks_ago = now_naive - timedelta(days=14)

        # 1. Telematics Fault Code Penalty
        active_codes = (
            self.db.query(DiagnosticCode)
            .filter(
                DiagnosticCode.unit_number == unit_number,
                DiagnosticCode.timestamp >= two_weeks_ago,
            )
            .all()
        )

        dtc_risk = 0.0
        critical_codes_count = 0
        for code in active_codes:
            sev = (code.severity or "").upper()
            if sev == "CRITICAL":
                dtc_risk += 0.35
                critical_codes_count += 1
            elif sev == "MEDIUM":
                dtc_risk += 0.15
            else:
                dtc_risk += 0.05
        dtc_risk = min(0.60, dtc_risk)

        # 2. Driver DVIR Defect Penalty
        recent_dvirs = (
            self.db.query(DVIRLog)
            .filter(
                DVIRLog.unit_number == unit_number,
                DVIRLog.log_date >= two_weeks_ago,
                DVIRLog.has_defects == True,
            )
            .order_by(DVIRLog.log_date.desc())
            .all()
        )

        dvir_risk = 0.0
        primary_concern = "NONE"
        if recent_dvirs:
            latest_dvir = recent_dvirs[0]
            dvir_risk = min(0.50, float(latest_dvir.nlp_severity_score or 0.0))
            primary_concern = latest_dvir.parsed_subsystem or "GENERAL"

        # 3. Odometer Mechanical Wear Penalty
        # Commercial trucks begin experiencing systemic wear past 300,000 miles
        odometer = float(asset.odometer_miles or 0.0)
        odometer_risk = min(0.15, max(0.0, (odometer - 250000.0) / 1000000.0))

        # Composite failure probability (0.0 to 1.0)
        composite_risk = min(1.0, dtc_risk + dvir_risk + odometer_risk)
        health_score = max(0.0, round(100.0 - (composite_risk * 100.0), 1))

        # Update persistent vehicle status
        asset.health_score = health_score
        if composite_risk >= 0.70 or critical_codes_count >= 1:
            asset.status = "MAINTENANCE_REQUIRED"
            action = "PULL_FROM_SERVICE_AND_REPAIR"
        elif composite_risk >= 0.35:
            asset.status = "ATTENTION_NEEDED"
            action = "SCHEDULE_INSPECTION_BEFORE_NEXT_DISPATCH"
        else:
            asset.status = "ACTIVE"
            action = "OPTIMAL_STATUS_PASS"

        self.db.commit()

        return {
            "unit_number": unit_number,
            "account_id": asset.account_id,
            "make_model": f"{asset.make} {asset.model} ({asset.year})",
            "odometer_miles": odometer,
            "health_score": health_score,
            "failure_risk_probability": round(composite_risk, 3),
            "primary_concern": primary_concern,
            "telematics_summary": {
                "active_dtc_count": len(active_codes),
                "critical_faults": critical_codes_count,
            },
            "status": asset.status,
            "recommended_action": action,
        }

    # ============================================================
    # 3. HAVERSINE GEO-ROUTING TO PREFERRED SERVICE HUBS
    # ============================================================

    def find_nearest_preferred_shop(
        self, unit_lat: float, unit_lon: float
    ) -> Optional[Dict[str, Any]]:
        """Calculates accurate spherical road distances using the Haversine formula

        to locate the optimal authorized maintenance facility.
        """
        centers = (
            self.db.query(ServiceCenter)
            .filter(ServiceCenter.is_preferred_partner == True)
            .all()
        )
        if not centers:
            return None

        def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            radius_miles = 3958.8
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(math.radians(lat1))
                * math.cos(math.radians(lat2))
                * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            return radius_miles * c

        ranked_centers = []
        for shop in centers:
            crow_miles = haversine_miles(
                unit_lat, unit_lon, float(shop.latitude), float(shop.longitude)
            )
            # 1.25 multiplier accounts for highway routing curvature
            driving_distance = round(crow_miles * 1.25, 1)
            ranked_centers.append((driving_distance, shop))

        ranked_centers.sort(key=lambda x: x[0])
        best_distance, best_shop = ranked_centers[0]

        return {
            "service_center_id": best_shop.id,
            "name": best_shop.name,
            "location": f"{best_shop.city}, {best_shop.state}",
            "driving_miles": best_distance,
            "hourly_labor_rate": f"${best_shop.hourly_labor_rate}/hr",
            "contact_phone": best_shop.contact_phone,
            "is_preferred_partner": best_shop.is_preferred_partner,
        }

    # ============================================================
    # 4. CROSS-ENGINE CASH FLOW WORK ORDER AUTOMATION
    # ============================================================

    def schedule_preventive_work_order(
        self,
        unit_number: str,
        subsystem: Optional[str] = None,
        estimated_cost: Optional[float] = None,
        days_ahead: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Schedules a maintenance work order and injects an OUTFLOW record

        into the Cash Flow Engine to protect forward liquidity reserves.
        """
        asset = (
            self.db.query(VehicleAsset)
            .filter(VehicleAsset.unit_number == unit_number)
            .first()
        )
        if not asset:
            raise ValueError(f"Vehicle asset '{unit_number}' does not exist.")

        # Determine subsystem parameters
        subsystem_key = (subsystem or "GENERAL").upper()
        profile = self.SUBSYSTEM_PROFILES.get(
            subsystem_key, self.SUBSYSTEM_PROFILES["GENERAL"]
        )

        final_cost = (
            estimated_cost
            if estimated_cost is not None
            else profile["default_repair_cost"]
        )
        lead_days = (
            days_ahead
            if days_ahead is not None
            else profile["turnaround_days"]
        )

        now_utc = datetime.now(timezone.utc)
        due_date = (now_utc + timedelta(days=lead_days)).replace(tzinfo=None)

        event_id = f"maint_{unit_number}_{int(now_utc.timestamp())}"

        # Write cash impact to the core ledger
        cash_outflow = CashEvent(
            event_id=event_id,
            account_id=asset.account_id or "acc1",
            source_type="MAINTENANCE",
            direction="OUTFLOW",
            category=f"Preventive Repair - {unit_number} ({subsystem_key})",
            gross_amount=round(final_cost, 2),
            net_amount=round(final_cost, 2),
            fee_deductions=0.0,
            status="PENDING",
            due_date=due_date,
            raw_payload=None,
        )

        # Flag the asset as undergoing scheduled maintenance
        asset.status = "MAINTENANCE_SCHEDULED"

        self.db.add(cash_outflow)
        self.db.commit()

        return {
            "work_order_id": event_id,
            "unit_number": unit_number,
            "account_id": asset.account_id,
            "subsystem": subsystem_key,
            "estimated_cost": final_cost,
            "scheduled_service_date": due_date.strftime("%b %d, %Y"),
            "cash_flow_engine_notified": True,
            "status": "SCHEDULED",
        }

    # ============================================================
    # 5. FLEET-WIDE HEALTH AUDIT
    # ============================================================

    def assess_entire_fleet(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Audits every vehicle asset in the specified fleet account

        and ranks them by failure risk descending.
        """
        query = self.db.query(VehicleAsset)
        if account_id and account_id != "all":
            query = query.filter(VehicleAsset.account_id == account_id)

        vehicles = query.all()
        results = [self.assess_vehicle_health(v.unit_number) for v in vehicles]
        results.sort(key=lambda x: x.get("failure_risk_probability", 0.0), reverse=True)
        return results