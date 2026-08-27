"""
Thin wrapper around the real Razorpay Python SDK, used in TEST MODE.

This is the part of the project that is a genuine, working Razorpay
integration (not simulated): creating an Order and verifying the payment
signature after Checkout.js completes. Get your test keys from
Dashboard -> Settings -> API Keys (make sure "Test Mode" is toggled on).

Docs: https://razorpay.com/docs/payments/server-integration/python/payment-gateway/build-integration/
"""
import hmac
import hashlib
import os
from typing import Dict

import razorpay

RAZORPAY_KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "")

_client = None


def get_client() -> razorpay.Client:
    global _client
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set. "
            "Copy .env.example to .env and paste your TEST mode keys."
        )
    if _client is None:
        _client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
    return _client


def create_authorization_order(amount_rupees: float, receipt: str, notes: Dict = None) -> Dict:
    """
    Creates a real Razorpay Order in test mode. This is what the user's
    browser will actually pay against (in test mode, with test card/UPI)
    to authorize the spending mandate.
    """
    client = get_client()
    order = client.order.create({
        "amount": int(round(amount_rupees * 100)),  # paise
        "currency": "INR",
        "receipt": receipt,
        "notes": notes or {},
        "payment_capture": 1,
    })
    return order


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verifies the HMAC-SHA256 signature Razorpay sends back after Checkout.js
    completes. NEVER trust a frontend "payment succeeded" callback without
    this -- it's the single most common corner people cut, and it's the
    difference between a real integration and a fake one.
    """
    if not RAZORPAY_KEY_SECRET:
        raise RuntimeError("RAZORPAY_KEY_SECRET not set.")
    payload = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
