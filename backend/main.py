import logging
import os
import uuid
from pathlib import Path
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

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
#
# session["keys"] holds any API keys THIS BROWSER SESSION supplied via the
# "no .env keys found" popup -- kept ONLY in this in-memory dict, for the
# life of the process. Never written to disk, never logged (see the
# _mask() helper below for what's safe to echo back), and never read from
# anywhere except the two call sites in this file that need them. If the
# server already has its own .env keys, sessions never need to supply
# anything and this dict just stays empty.
SESSIONS: Dict[str, Dict] = {}


def _session(session_id: str) -> Dict:
    return SESSIONS.setdefault(session_id, {
        "cart": [], "mandate_id": None, "last_shown_items": [],
        "keys": {"gemini_api_key": None, "razorpay_key_id": None, "razorpay_key_secret": None},
    })


def _mask(secret: Optional[str]) -> Optional[str]:
    """For display only -- never return a full key in any API response."""
    if not secret:
        return None
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}"


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
    try:
        with agent_mod.gemini_key_override(session["keys"].get("gemini_api_key")):
            result = agent_mod.run_agent(
                req.message, session["cart"], session["mandate_id"], session.get("last_shown_items", [])
            )
    except agent_mod.AgentUnavailableError as e:
        # No silent rule-based/template degrade -- surface a clear, honest
        # error instead. Nothing in session state (cart, mandate) has been
        # touched at this point, since node_interpret is the very first
        # node in the graph and raises before any action node can run.
        raise HTTPException(status_code=503, detail=f"AI is currently unavailable: {e}")
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
    # compose_reply_from_facts() never raises for facts.type == "checkout" --
    # if Gemini is unavailable it falls back to a minimal, honestly-labeled
    # factual notice instead (see agent._checkout_system_notice), because
    # the debit outcome above may already be real and can never be hidden
    # behind an error just because the reply-writer is down.
    with agent_mod.gemini_key_override(session["keys"].get("gemini_api_key")):
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
    the amount the user wants to block, then have the frontend open
    Checkout.js against it. Requires real Razorpay test-mode keys -- either
    from the server's .env or from this session's API keys prompt. If
    neither is configured, this fails with a clear 400 explaining exactly
    that, instead of silently creating a fake order.
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

    session = _session(req.session_id)
    keys = session["keys"]

    try:
        order = razorpay_client.create_authorization_order(
            amount_rupees=req.max_amount,
            receipt=f"mandate_{uuid.uuid4().hex[:10]}",
            notes={"purpose": "agentic_checkout_mandate_authorization"},
            key_id=keys.get("razorpay_key_id"),
            key_secret=keys.get("razorpay_key_secret"),
        )
    except RuntimeError as e:
        # Missing/unset API keys. 400, not 500 -- this is a configuration
        # problem the user can fix themselves via the API keys prompt, not
        # a server fault.
        raise HTTPException(status_code=400, detail=str(e))
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
        "razorpay_key_id": keys.get("razorpay_key_id") or razorpay_client.RAZORPAY_KEY_ID,
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
    session = _session(req.session_id)
    try:
        ok = razorpay_client.verify_payment_signature(
            req.razorpay_order_id, req.razorpay_payment_id, req.razorpay_signature,
            key_secret=session["keys"].get("razorpay_key_secret"),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=400, detail="Signature verification failed.")

    m = mandate_mod.store.authorize(req.mandate_id, req.razorpay_payment_id)
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


# ---------------------------------------------------------------------------
# API key configuration -- "no .env keys? ask in the browser instead" flow.
#
# Security notes:
#  - Keys supplied here live ONLY in the in-memory SESSIONS dict (this
#    process's RAM), for this session, for the life of the process. They
#    are never written to disk, never included in any log line, and never
#    echoed back in full in any response (see _mask() above).
#  - There is no "skip -> simulate everything" mode anymore. If a key is
#    missing, the relevant endpoint fails with a clear, actionable error
#    (see AgentUnavailableError handling in /api/chat, and the plain
#    RuntimeError -> HTTPException(400) handling in /api/mandate/create
#    and /api/mandate/verify) instead of silently substituting fake
#    behavior. The popup just lets a session add real keys without
#    touching the server's .env.
# ---------------------------------------------------------------------------

@app.get("/api/config/status")
def config_status(session_id: str):
    session = _session(session_id)
    keys = session["keys"]
    return {
        # Does the SERVER already have these via .env? If both are true,
        # the frontend never needs to show the popup at all.
        "gemini_env_configured": bool(agent_mod.GEMINI_API_KEY),
        "razorpay_env_configured": bool(razorpay_client.RAZORPAY_KEY_ID and razorpay_client.RAZORPAY_KEY_SECRET),
        # Has THIS session already supplied its own key(s)?
        "gemini_session_configured": bool(keys.get("gemini_api_key")),
        "razorpay_session_configured": bool(keys.get("razorpay_key_id") and keys.get("razorpay_key_secret")),
        "gemini_key_preview": _mask(keys.get("gemini_api_key")),
        "razorpay_key_id_preview": _mask(keys.get("razorpay_key_id")),
    }


class ConfigKeysRequest(BaseModel):
    session_id: str
    gemini_api_key: Optional[str] = None
    razorpay_key_id: Optional[str] = None
    razorpay_key_secret: Optional[str] = None


@app.post("/api/config/keys")
def set_config_keys(req: ConfigKeysRequest):
    session = _session(req.session_id)
    if req.gemini_api_key:
        session["keys"]["gemini_api_key"] = req.gemini_api_key.strip()
    if req.razorpay_key_id:
        session["keys"]["razorpay_key_id"] = req.razorpay_key_id.strip()
    if req.razorpay_key_secret:
        session["keys"]["razorpay_key_secret"] = req.razorpay_key_secret.strip()
    return {
        "gemini_session_configured": bool(session["keys"]["gemini_api_key"]),
        "razorpay_session_configured": bool(session["keys"]["razorpay_key_id"] and session["keys"]["razorpay_key_secret"]),
    }


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


# ---------------------------------------------------------------------------
# Serve the project's own docs at the paths the landing page links to
# (README.md, CHALLENGES.md), read live from the project root rather than
# duplicated into frontend/ -- one source of truth, no risk of the copy
# going stale. These are real @app.get routes, so (per the ordering note
# below) they're matched before the StaticFiles mount ever sees the request,
# regardless of where they're defined relative to it.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@app.get("/README.md")
def readme():
    path = PROJECT_ROOT / "README.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="README.md not found.")
    return Response(content=path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


@app.get("/CHALLENGES.md")
def challenges():
    path = PROJECT_ROOT / "CHALLENGES.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="CHALLENGES.md not found.")
    return Response(content=path.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")


# ---------------------------------------------------------------------------
# Serve the frontend from this same process.
#
# On a single-service host like Render, there's only one process and one
# port -- there's no separate static file server, so uvicorn has to serve
# BOTH the API above and the plain HTML/CSS/JS frontend. This mount MUST be
# the last thing registered: FastAPI/Starlette checks routes in the order
# they were added, so every @app.get/@app.post route above still gets
# matched first (e.g. GET /api/health, GET /README.md never fall through to
# here) and this mount only ever catches whatever's left -- "/" ->
# frontend/index.html, "/landing.html" -> frontend/landing.html, etc.
# html=True also makes it serve index.html for any unmatched path, so the
# app still loads correctly even if someone deep-links to a path that
# doesn't exist as a file.
# ---------------------------------------------------------------------------
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logging.getLogger("main").warning("frontend/ directory not found at %s -- static files not served.", FRONTEND_DIR)
