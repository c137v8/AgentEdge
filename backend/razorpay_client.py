"""
Thin wrapper around the real Razorpay Python SDK, used in TEST MODE.

This is the part of the project that is a genuine, working Razorpay
integration (not simulated): creating an Order and verifying the payment
signature after Checkout.js completes. Get your test keys from
Dashboard -> Settings -> API Keys (make sure "Test Mode" is toggled on).

Docs: https://razorpay.com/docs/payments/server-integration/python/payment-gateway/build-integration/

Keys can come from two places:
  1. The server's own .env (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET below) --
     used for every session unless overridden.
  2. A key pair a specific browser session typed into the "no .env keys
     found" popup (see main.py's /api/config/keys). Those are passed in
     explicitly per call below and are NEVER read from or written to disk,
     logged, or cached in a module-level variable -- they live only in
     main.py's in-memory SESSIONS dict for the lifetime of the process.

If NEITHER is available, there is no silent fallback: get_client() below
raises a clear RuntimeError, which main.py turns into a plain HTTP error
the frontend shows to the user. No mock order, no simulated payment --
just an honest "you need to add a key for this" instead of quietly
pretending to have processed a real authorization.
"""
import hmac
import hashlib
import os
from typing import Dict, Optional

import razorpay

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")


def get_client(key_id: Optional[str] = None, key_secret: Optional[str] = None) -> razorpay.Client:
    key_id = key_id or RAZORPAY_KEY_ID
    key_secret = key_secret or RAZORPAY_KEY_SECRET
    if not key_id or not key_secret:
        raise RuntimeError(
            "Razorpay key id/secret not set for this session. Paste your TEST mode "
            "keys into the API keys prompt, or set RAZORPAY_KEY_ID / "
            "RAZORPAY_KEY_SECRET in .env on the server."
        )
    # A fresh Client() per call is intentional: key pairs can differ across
    # sessions, so there is no single global client to safely cache anymore.
    # The SDK's Client constructor does no network I/O, so this is cheap.
    return razorpay.Client(auth=(key_id, key_secret))


def create_authorization_order(amount_rupees: float, receipt: str, notes: Dict = None,
                                key_id: Optional[str] = None, key_secret: Optional[str] = None) -> Dict:
    """
    Creates a real Razorpay Order in test mode. This is what the user's
    browser will actually pay against (in test mode, with test card/UPI)
    to authorize the spending mandate.
    """
    client = get_client(key_id, key_secret)
    order = client.order.create({
        "amount": int(round(amount_rupees * 100)),  # paise
        "currency": "INR",
        "receipt": receipt,
        "notes": notes or {},
        "payment_capture": 1,
    })
    return order


def verify_payment_signature(order_id: str, payment_id: str, signature: str,
                              key_secret: Optional[str] = None) -> bool:
    """
    Verifies the HMAC-SHA256 signature Razorpay sends back after Checkout.js
    completes. NEVER trust a frontend "payment succeeded" callback without
    this -- it's the single most common corner people cut, and it's the
    difference between a real integration and a fake one.
    """
    secret = key_secret or RAZORPAY_KEY_SECRET
    if not secret:
        raise RuntimeError("Razorpay key secret not set for this session.")
    payload = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)

