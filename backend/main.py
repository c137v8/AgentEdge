import os
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent as agent_mod
import inventory
import mandate as mandate_mod
import razorpay_client

app = FastAPI(title="Agentic Checkout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Naive per-session cart store, keyed by a session id the frontend generates.
# Swap for Redis in a real deployment.
SESSIONS: Dict[str, Dict] = {}


def _session(session_id: str) -> Dict:
    return SESSIONS.setdefault(session_id, {"cart": [], "mandate_id": None})


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

@app.get("/api/products")
def list_products(q: str = "", max_price: Optional[int] = None):
    return {"items": inventory.search(q, max_price=max_price)}


# ---------------------------------------------------------------------------
# Chat (the agent)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = _session(req.session_id)
    result = agent_mod.run_agent(req.message, session["cart"], session["mandate_id"])
    session["cart"] = result.get("cart", session["cart"])
    return {
        "reply": result.get("reply", ""),
        "cart": session["cart"],
        "cart_total": sum(i["price"] for i in session["cart"]),
        "action": result.get("action_result"),
    }


# ---------------------------------------------------------------------------
# Mandate authorization (REAL Razorpay test-mode order creation)
# ---------------------------------------------------------------------------

class CreateMandateRequest(BaseModel):
    session_id: str
    max_amount: float          # rupees, total the user is willing to block
    per_txn_cap: float         # rupees, guardrail per single AI purchase
    days_valid: int = 30


@app.post("/api/mandate/create")
def create_mandate(req: CreateMandateRequest):
    """
    Step 1 of authorization: create a real Razorpay Order (test mode) for
    the amount the user wants to block. The frontend opens Checkout.js
    against this order. This is a genuine Razorpay API call, not mocked.
    """
    try:
        order = razorpay_client.create_authorization_order(
            amount_rupees=req.max_amount,
            receipt=f"mandate_{uuid.uuid4().hex[:10]}",
            notes={"purpose": "agentic_checkout_mandate_authorization"},
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    m = mandate_mod.store.create(
        user_id=req.session_id,
        max_amount_rupees=req.max_amount,
        per_txn_cap_rupees=req.per_txn_cap,
        razorpay_order_id=order["id"],
        days_valid=req.days_valid,
    )
    return {
        "mandate_id": m.id,
        "razorpay_order_id": order["id"],
        "razorpay_key_id": razorpay_client.RAZORPAY_KEY_ID,
        "amount_paise": order["amount"],
        "currency": order["currency"],
    }


class VerifyMandateRequest(BaseModel):
    session_id: str
    mandate_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


@app.post("/api/mandate/verify")
def verify_mandate(req: VerifyMandateRequest):
    """
    Step 2 of authorization: verify the signature Razorpay returns after
    Checkout.js completes, then flip the mandate to AUTHORIZED. From this
    point on, the agent can debit against it with zero further prompts,
    up to the guardrails set at creation time.
    """
    ok = razorpay_client.verify_payment_signature(
        req.razorpay_order_id, req.razorpay_payment_id, req.razorpay_signature
    )
    if not ok:
        raise HTTPException(status_code=400, detail="Signature verification failed.")

    m = mandate_mod.store.authorize(req.mandate_id, req.razorpay_payment_id)
    session = _session(req.session_id)
    session["mandate_id"] = m.id
    return {"status": m.status.value, "mandate": m.to_public_dict()}


@app.get("/api/mandate/{mandate_id}")
def get_mandate(mandate_id: str):
    m = mandate_mod.store.get(mandate_id)
    if not m:
        raise HTTPException(status_code=404, detail="Not found")
    return m.to_public_dict()


class RevokeRequest(BaseModel):
    session_id: str
    mandate_id: str


@app.post("/api/mandate/revoke")
def revoke_mandate(req: RevokeRequest):
    m = mandate_mod.store.revoke(req.mandate_id)
    session = _session(req.session_id)
    session["mandate_id"] = None
    return {"status": m.status.value}


@app.get("/api/health")
def health():
    return {"ok": True, "razorpay_configured": bool(razorpay_client.RAZORPAY_KEY_ID)}
