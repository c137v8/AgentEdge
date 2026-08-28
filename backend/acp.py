"""
Agentic Commerce Protocol (ACP) -- checkout-session engine.

ACP is the open standard co-developed by OpenAI and Stripe (Apache 2.0,
date-versioned specs) for how an AI agent, a buyer, and a merchant
complete a purchase while the merchant stays the system of record for
orders, payments, tax, and compliance:
https://github.com/agentic-commerce-protocol/agentic-commerce-protocol

This module implements the "Agentic Checkout Specification" (ACS) part
of ACP -- the CheckoutSession resource and its five REST operations --
as the transactional/execution layer of this app:

    create   POST /checkout_sessions
    update   POST /checkout_sessions/{id}
    retrieve GET  /checkout_sessions/{id}
    complete POST /checkout_sessions/{id}/complete
    cancel   POST /checkout_sessions/{id}/cancel

(wired up as real FastAPI routes in main.py, and also used internally by
agent.execute_checkout so the chat-driven "Confirm & Pay" button goes
through the same ACP session lifecycle as an external agent would.)

What's real vs. simplified here, in the same spirit as README.md's
"what's real vs simulated" table:
  - The CheckoutSession state machine, line items, totals-in-minor-units,
    and the {type, code, message, param} error shape all follow the
    spec.
  - Payment itself is NOT delegated to Stripe's Shared Payment Token
    network (that requires a real Stripe/OpenAI merchant integration).
    Instead this merchant declares a single accepted payment handler of
    its own -- "reserve_pay_mandate" -- a "seller-backed payment
    handler" (a pattern ACP explicitly supports, see
    rfc.seller_backed_payment_handler.md) backed by the Reserve Pay
    mandate engine in mandate.py. complete_session() below is the ONLY
    function that hands a payment_data.token to mandate.store.debit().
  - No webhook delivery (order_create/order_update) and no request
    signing -- both real ACP requirements this demo doesn't implement.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mandate as mandate_mod
import merchant as merchant_mod

API_VERSION = merchant_mod.MERCHANT["acp_version"]


class ACPError(Exception):
    """
    Mirrors ACP's flat error object:
      {"type": "invalid_request", "code": "...", "message": "...", "param": "$.items[0].id"}
    type is one of invalid_request | request_not_idempotent | processing_error | service_unavailable.
    """

    def __init__(self, type_: str, code: str, message: str, param: Optional[str] = None, http_status: int = 400):
        self.type = type_
        self.code = code
        self.message = message
        self.param = param
        self.http_status = http_status
        super().__init__(message)

    def to_dict(self) -> Dict:
        d = {"type": self.type, "code": self.code, "message": self.message}
        if self.param:
            d["param"] = self.param
        return d


@dataclass
class CheckoutSession:
    id: str
    status: str  # not_ready_for_payment | ready_for_payment | completed | canceled
    currency: str
    line_items: List[Dict] = field(default_factory=list)
    fulfillment_address: Optional[Dict] = None
    buyer: Optional[Dict] = None
    order: Optional[Dict] = None
    mandate_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def totals(self) -> List[Dict]:
        items_total = sum(li["total"] for li in self.line_items)
        return [
            {"type": "items_base_amount", "display_text": "Items", "amount": items_total},
            {"type": "tax", "display_text": "Tax", "amount": 0},
            {"type": "total", "display_text": "Total", "amount": items_total},
        ]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "status": self.status,
            "currency": self.currency,
            "line_items": self.line_items,
            "totals": self.totals(),
            "fulfillment_address": self.fulfillment_address,
            "fulfillment_options": [
                {"type": "digital_or_pickup", "id": "instant", "title": "Instant (no shipping)", "subtotal": 0, "tax": 0, "total": 0}
            ],
            "buyer": self.buyer,
            "messages": [],
            "links": [{"type": "terms_of_use", "url": merchant_mod.MERCHANT["url"] + "/terms"}],
            "merchant": {
                "id": merchant_mod.MERCHANT["merchant_id"],
                "name": merchant_mod.MERCHANT["storefront_name"],
                "support_email": merchant_mod.MERCHANT["support_email"],
            },
            "payment_provider": {
                "provider": "reserve_pay_mandate",
                "supported_payment_methods": ["seller_backed"],
            },
            "order": self.order,
        }


# In-memory session store + idempotency cache. Swap for Redis/Postgres in
# a real deployment (same caveat as mandate.MandateStore and main.SESSIONS).
_SESSIONS: Dict[str, CheckoutSession] = {}
_IDEMPOTENCY: Dict[str, CheckoutSession] = {}


def _line_items_from_cart(cart_items: List[Dict]) -> List[Dict]:
    """Amounts are integers in minor units (paise), per spec ("Amounts MUST be integers in minor units")."""
    line_items = []
    for it in cart_items:
        base_amount = int(round(it["price"] * 100))
        line_items.append({
            "id": f"li_{uuid.uuid4().hex[:8]}",
            "item": {"id": it["id"], "name": it["name"]},
            "quantity": 1,
            "base_amount": base_amount,
            "discount": 0,
            "subtotal": base_amount,
            "tax": 0,
            "total": base_amount,
        })
    return line_items


def create_session(cart_items: List[Dict], buyer: Optional[Dict] = None,
                    fulfillment_address: Optional[Dict] = None,
                    mandate_id: Optional[str] = None,
                    idempotency_key: Optional[str] = None) -> CheckoutSession:
    if idempotency_key and idempotency_key in _IDEMPOTENCY:
        return _IDEMPOTENCY[idempotency_key]
    if not cart_items:
        raise ACPError("invalid_request", "empty_cart", "Cannot create a checkout session with no items.", param="$.items")

    session = CheckoutSession(
        id=f"checkout_session_{uuid.uuid4().hex[:16]}",
        status="ready_for_payment" if mandate_id else "not_ready_for_payment",
        currency="inr",
        line_items=_line_items_from_cart(cart_items),
        fulfillment_address=fulfillment_address,
        buyer=buyer,
        mandate_id=mandate_id,
    )
    _SESSIONS[session.id] = session
    if idempotency_key:
        _IDEMPOTENCY[idempotency_key] = session
    return session


def get_session(session_id: str) -> CheckoutSession:
    s = _SESSIONS.get(session_id)
    if not s:
        raise ACPError("invalid_request", "not_found", "No such checkout session.", param="$.id", http_status=404)
    return s


def update_session(session_id: str, buyer: Optional[Dict] = None,
                    fulfillment_address: Optional[Dict] = None) -> CheckoutSession:
    s = get_session(session_id)
    if s.status in ("completed", "canceled"):
        raise ACPError("invalid_request", "session_not_updatable", f"Session is {s.status} and cannot be updated.")
    if buyer:
        s.buyer = buyer
    if fulfillment_address:
        s.fulfillment_address = fulfillment_address
    return s


def cancel_session(session_id: str) -> CheckoutSession:
    s = get_session(session_id)
    if s.status == "completed":
        raise ACPError("invalid_request", "already_completed", "Cannot cancel a completed session.")
    s.status = "canceled"
    return s


def complete_session(session_id: str, payment_data: Dict, buyer: Optional[Dict] = None,
                      idempotency_key: Optional[str] = None) -> CheckoutSession:
    """
    The ONLY function in this module (and, by extension, the ONLY code
    path reachable from the ACP REST surface) that can call
    mandate.store.debit(). Mirrors the same "checkout preview vs.
    execute" separation used elsewhere in this project: create_session()
    never moves money, only complete_session() can, and only once a
    payment_data.token for the merchant's declared handler is supplied.
    """
    if idempotency_key and idempotency_key in _IDEMPOTENCY:
        return _IDEMPOTENCY[idempotency_key]

    s = get_session(session_id)
    if s.status == "completed":
        raise ACPError("invalid_request", "already_completed", "This session was already completed.")
    if s.status == "canceled":
        raise ACPError("invalid_request", "session_canceled", "This session has been canceled and can't be paid.")
    if buyer:
        s.buyer = buyer

    provider = (payment_data or {}).get("provider")
    token = (payment_data or {}).get("token")
    accepted_ids = {h["id"] for h in merchant_mod.MERCHANT["accepted_payment_handlers"]}
    if provider not in accepted_ids or not token:
        raise ACPError(
            "invalid_request", "unsupported_payment_handler",
            f"{merchant_mod.MERCHANT['storefront_name']} only accepts these payment handlers: {sorted(accepted_ids)}.",
            param="$.payment_data.provider",
        )

    total_paise = sum(li["total"] for li in s.line_items)
    amount_rupees = total_paise / 100
    idem = idempotency_key or f"acp_complete_{s.id}"

    try:
        # token IS the mandate id -- our "delegated payment token" is
        # just a reference to the Reserve Pay mandate the user already
        # authorized (see mandate.py). A real Stripe-backed handler would
        # exchange this for a Shared Payment Token instead.
        entry = mandate_mod.store.debit(
            mandate_id=token,
            amount_rupees=amount_rupees,
            reason=f"ACP order: {', '.join(li['item']['name'] for li in s.line_items)}",
            idempotency_key=idem,
        )
    except mandate_mod.DebitError as e:
        # Map our own guardrail errors onto ACP's processing_error type.
        raise ACPError("processing_error", e.code, e.message)

    s.status = "completed"
    s.order = {
        "id": f"order_{uuid.uuid4().hex[:12]}",
        "checkout_session_id": s.id,
        "status": "created",
        "permalink_url": f"{merchant_mod.MERCHANT['url']}/orders/{entry.id}",
    }
    if idempotency_key:
        _IDEMPOTENCY[idempotency_key] = s
    return s
