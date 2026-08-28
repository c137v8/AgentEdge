import logging
import os
import uuid
from typing import Dict, List, Optional

# MUST run before importing agent/razorpay_client -- both read env vars
# (GEMINI_API_KEY, RAZORPAY_KEY_ID, etc.) at module import time via
# os.environ.get(...). If .env isn't loaded first, those reads silently
# return "" and everything downstream falls back / 500s with no obvious
# cause. This was the root cause of both the mandate 500 and the LLM
# never firing -- python-dotenv was in requirements.txt but never
# actually invoked anywhere.
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import acp
import agent as agent_mod
import inventory
import mandate as mandate_mod
import merchant as merchant_mod
import razorpay_client

app = FastAPI(title="Agentic Checkout API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_acp_version_header(request: Request, call_next):
    """Every ACP response carries the spec version this merchant implements."""
    response = await call_next(request)
    if request.url.path.startswith("/checkout_sessions"):
        response.headers["API-Version"] = acp.API_VERSION
    return response

# Naive per-session cart store, keyed by a session id the frontend generates.
# Swap for Redis in a real deployment.
SESSIONS: Dict[str, Dict] = {}


def _session(session_id: str) -> Dict:
    return SESSIONS.setdefault(session_id, {"cart": [], "mandate_id": None, "last_shown_items": []})


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
    result = agent_mod.run_agent(
        req.message, session["cart"], session["mandate_id"], session.get("last_shown_items", [])
    )
    session["cart"] = result.get("cart", session["cart"])
    session["last_shown_items"] = result.get("last_shown_items", session.get("last_shown_items", []))
    return {
        "reply": result.get("reply", ""),
        "cart": session["cart"],
        "cart_total": sum(i["price"] for i in session["cart"]),
        "action": result.get("action_result"),
    }


# ---------------------------------------------------------------------------
# Checkout confirmation -- the ONLY endpoint that can actually spend money.
# The chat endpoint above can only ever return a "confirm_checkout" PREVIEW
# (see agent.node_checkout); nothing is charged until the user explicitly
# hits the "Confirm & Pay" button in the frontend, which calls this.
# ---------------------------------------------------------------------------

class ConfirmCheckoutRequest(BaseModel):
    session_id: str


@app.post("/api/checkout/confirm")
def confirm_checkout(req: ConfirmCheckoutRequest):
    session = _session(req.session_id)
    facts = agent_mod.execute_checkout(session["cart"], session["mandate_id"])
    if facts.get("ok"):
        session["cart"] = []
    reply = agent_mod.compose_reply_from_facts(facts)
    return {
        "reply": reply,
        "cart": session["cart"],
        "cart_total": sum(i["price"] for i in session["cart"]),
        "action": facts,
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
    # Fail with a clear 400 instead of an opaque 500 if the request itself
    # is invalid (e.g. above Reserve Pay's real Rs 10,000 block cap) --
    # this check is cheap and doesn't require calling Razorpay at all.
    if req.max_amount <= 0 or req.per_txn_cap <= 0:
        raise HTTPException(status_code=400, detail="max_amount and per_txn_cap must be positive.")
    if req.max_amount > mandate_mod.MAX_BLOCK_AMOUNT_PAISE / 100:
        raise HTTPException(
            status_code=400,
            detail=f"Reserve Pay caps blocks at Rs {mandate_mod.MAX_BLOCK_AMOUNT_PAISE/100:.0f}. Lower max_amount.",
        )
    if req.per_txn_cap > req.max_amount:
        raise HTTPException(status_code=400, detail="per_txn_cap cannot exceed max_amount.")

    try:
        order = razorpay_client.create_authorization_order(
            amount_rupees=req.max_amount,
            receipt=f"mandate_{uuid.uuid4().hex[:10]}",
            notes={"purpose": "agentic_checkout_mandate_authorization"},
        )
    except RuntimeError as e:
        # Missing/unset API keys -- most common cause of a 500 here.
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Razorpay SDK errors (bad key format, invalid request, network
        # issue, etc). Surfacing the real message instead of a blank 500
        # is the difference between debugging in 10 seconds vs 10 minutes.
        raise HTTPException(status_code=502, detail=f"Razorpay order creation failed: {e}")

    try:
        m = mandate_mod.store.create(
            user_id=req.session_id,
            max_amount_rupees=req.max_amount,
            per_txn_cap_rupees=req.per_txn_cap,
            razorpay_order_id=order["id"],
            days_valid=req.days_valid,
        )
    except mandate_mod.DebitError as e:
        raise HTTPException(status_code=400, detail=e.message)

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


@app.get("/api/merchant")
def get_merchant():
    """Dummy merchant profile the frontend can display alongside the Confirm & Pay card."""
    return merchant_mod.MERCHANT


# ---------------------------------------------------------------------------
# Agentic Commerce Protocol (ACP) -- checkout-session REST surface.
#
# These five routes are the real transactional/execution layer: an
# ACP-speaking agent (this app's own agent.py, or in principle any other
# ACP client) creates a session from cart items, optionally updates it,
# and completes it with a payment_data token to actually move money.
# Nothing here bypasses the mandate guardrails -- acp.complete_session()
# is the only function that can call mandate.store.debit() (see acp.py).
# ---------------------------------------------------------------------------

class ACPLineItemIn(BaseModel):
    id: str
    quantity: int = 1


class ACPBuyerIn(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None


class ACPAddressIn(BaseModel):
    name: Optional[str] = None
    line_one: Optional[str] = None
    line_two: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class CreateCheckoutSessionRequest(BaseModel):
    items: List[ACPLineItemIn]
    buyer: Optional[ACPBuyerIn] = None
    fulfillment_address: Optional[ACPAddressIn] = None


def _resolve_acp_items(items: List[ACPLineItemIn]) -> List[Dict]:
    resolved: List[Dict] = []
    for li in items:
        product = inventory.get_by_id(li.id)
        if not product:
            raise HTTPException(
                status_code=400,
                detail=acp.ACPError("invalid_request", "item_not_found", f"No such item '{li.id}'.", param="$.items[].id").to_dict(),
            )
        resolved.extend([product] * max(1, li.quantity))
    return resolved


@app.post("/checkout_sessions")
def acp_create_checkout_session(req: CreateCheckoutSessionRequest, idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    resolved_items = _resolve_acp_items(req.items)
    try:
        session = acp.create_session(
            resolved_items,
            buyer=req.buyer.dict(exclude_none=True) if req.buyer else None,
            fulfillment_address=req.fulfillment_address.dict(exclude_none=True) if req.fulfillment_address else None,
            idempotency_key=idempotency_key,
        )
    except acp.ACPError as e:
        raise HTTPException(status_code=e.http_status, detail=e.to_dict())
    return session.to_dict()


class UpdateCheckoutSessionRequest(BaseModel):
    buyer: Optional[ACPBuyerIn] = None
    fulfillment_address: Optional[ACPAddressIn] = None


@app.post("/checkout_sessions/{checkout_session_id}")
def acp_update_checkout_session(checkout_session_id: str, req: UpdateCheckoutSessionRequest):
    try:
        session = acp.update_session(
            checkout_session_id,
            buyer=req.buyer.dict(exclude_none=True) if req.buyer else None,
            fulfillment_address=req.fulfillment_address.dict(exclude_none=True) if req.fulfillment_address else None,
        )
    except acp.ACPError as e:
        raise HTTPException(status_code=e.http_status, detail=e.to_dict())
    return session.to_dict()


@app.get("/checkout_sessions/{checkout_session_id}")
def acp_get_checkout_session(checkout_session_id: str):
    try:
        session = acp.get_session(checkout_session_id)
    except acp.ACPError as e:
        raise HTTPException(status_code=e.http_status, detail=e.to_dict())
    return session.to_dict()


class ACPPaymentDataIn(BaseModel):
    token: str
    provider: str
    billing_address: Optional[ACPAddressIn] = None


class CompleteCheckoutSessionRequest(BaseModel):
    buyer: Optional[ACPBuyerIn] = None
    payment_data: ACPPaymentDataIn


@app.post("/checkout_sessions/{checkout_session_id}/complete")
def acp_complete_checkout_session(checkout_session_id: str, req: CompleteCheckoutSessionRequest,
                                   idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    try:
        session = acp.complete_session(
            checkout_session_id,
            payment_data=req.payment_data.dict(exclude_none=True),
            buyer=req.buyer.dict(exclude_none=True) if req.buyer else None,
            idempotency_key=idempotency_key,
        )
    except acp.ACPError as e:
        raise HTTPException(status_code=e.http_status, detail=e.to_dict())
    return session.to_dict()


@app.post("/checkout_sessions/{checkout_session_id}/cancel")
def acp_cancel_checkout_session(checkout_session_id: str):
    try:
        session = acp.cancel_session(checkout_session_id)
    except acp.ACPError as e:
        raise HTTPException(status_code=e.http_status, detail=e.to_dict())
    return session.to_dict()
