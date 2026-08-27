"""
Mandate engine: simulates the *post-authorization* behaviour of Razorpay's
UPI Reserve Pay (Single Block Multi Debit / SBMD).

WHY SIMULATED: SBMD requires Razorpay to activate Reserve Pay on a business
account (support-gated, not self-serve). A student project cannot get that
activation before a hackathon deadline. So:

  - The INITIAL block/authorization step is REAL Razorpay test-mode
    integration (Orders API + Checkout.js), see razorpay_client.py.
  - Everything AFTER authorization -- the agent debiting multiple times
    against that one blocked amount without further PIN prompts -- is
    simulated here, using the same state machine and guardrails a real
    SBMD integration would need (see Razorpay's subscription lifecycle:
    INITIALIZED -> AUTHORIZED -> charge -> refund).

This file is where the actual "AI Risk Manager" engineering lives:
idempotency, spend caps, per-transaction caps, expiry, and an audit ledger.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

MAX_BLOCK_AMOUNT_PAISE = 10_000_00   # Real SBMD cap: Rs 10,000 (see Razorpay docs)
MAX_MANDATE_DAYS = 90                 # Real SBMD cap: 90 days


class MandateStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    AUTHORIZED = "AUTHORIZED"
    EXHAUSTED = "EXHAUSTED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class DebitError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass
class LedgerEntry:
    id: str
    amount_paise: int
    reason: str
    status: str  # SUCCESS | BLOCKED | FAILED
    timestamp: float
    idempotency_key: str


@dataclass
class Mandate:
    id: str
    user_id: str
    max_amount_paise: int          # total blocked amount
    per_txn_cap_paise: int         # guardrail: max any single AI-initiated debit
    razorpay_order_id: Optional[str]
    razorpay_payment_id: Optional[str]
    created_at: float
    expires_at: float
    status: MandateStatus = MandateStatus.INITIALIZED
    utilised_paise: int = 0
    ledger: List[LedgerEntry] = field(default_factory=list)
    _seen_idempotency_keys: set = field(default_factory=set)

    @property
    def available_paise(self) -> int:
        return self.max_amount_paise - self.utilised_paise

    def to_public_dict(self) -> Dict:
        return {
            "mandate_id": self.id,
            "status": self.status.value,
            "max_amount": self.max_amount_paise / 100,
            "per_txn_cap": self.per_txn_cap_paise / 100,
            "utilised": self.utilised_paise / 100,
            "available": self.available_paise / 100,
            "expires_at": self.expires_at,
            "ledger": [
                {
                    "id": e.id,
                    "amount": e.amount_paise / 100,
                    "reason": e.reason,
                    "status": e.status,
                    "timestamp": e.timestamp,
                }
                for e in self.ledger
            ],
        }


class MandateStore:
    """In-memory store. Swap for Redis/Postgres in a real deployment."""

    def __init__(self):
        self._mandates: Dict[str, Mandate] = {}

    def create(self, user_id: str, max_amount_rupees: float, per_txn_cap_rupees: float,
               razorpay_order_id: str, days_valid: int = 90) -> Mandate:
        max_paise = int(round(max_amount_rupees * 100))
        if max_paise > MAX_BLOCK_AMOUNT_PAISE:
            raise DebitError("LIMIT_EXCEEDED", f"Reserve Pay caps blocks at Rs {MAX_BLOCK_AMOUNT_PAISE/100:.0f}")
        days_valid = min(days_valid, MAX_MANDATE_DAYS)
        m = Mandate(
            id=f"mandate_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            max_amount_paise=max_paise,
            per_txn_cap_paise=int(round(per_txn_cap_rupees * 100)),
            razorpay_order_id=razorpay_order_id,
            razorpay_payment_id=None,
            created_at=time.time(),
            expires_at=time.time() + days_valid * 86400,
        )
        self._mandates[m.id] = m
        return m

    def get(self, mandate_id: str) -> Optional[Mandate]:
        return self._mandates.get(mandate_id)

    def authorize(self, mandate_id: str, razorpay_payment_id: str) -> Mandate:
        m = self._require(mandate_id)
        m.status = MandateStatus.AUTHORIZED
        m.razorpay_payment_id = razorpay_payment_id
        return m

    def revoke(self, mandate_id: str) -> Mandate:
        m = self._require(mandate_id)
        m.status = MandateStatus.REVOKED
        return m

    def debit(self, mandate_id: str, amount_rupees: float, reason: str, idempotency_key: str) -> LedgerEntry:
        """
        The core guardrail chokepoint. Every AI-initiated purchase MUST pass
        through here. This is intentionally strict and fails closed.
        """
        m = self._require(mandate_id)
        amount_paise = int(round(amount_rupees * 100))

        # 1. Idempotency: never double-debit for the same logical action
        # (e.g. agent retries after a network timeout).
        if idempotency_key in m._seen_idempotency_keys:
            existing = next((e for e in m.ledger if e.idempotency_key == idempotency_key), None)
            if existing:
                return existing
            raise DebitError("DUPLICATE_REQUEST", "This action was already processed.")

        entry_id = f"txn_{uuid.uuid4().hex[:10]}"

        def _blocked(code: str, msg: str) -> LedgerEntry:
            entry = LedgerEntry(entry_id, amount_paise, reason, "BLOCKED", time.time(), idempotency_key)
            m.ledger.append(entry)
            m._seen_idempotency_keys.add(idempotency_key)
            raise DebitError(code, msg)

        # 2. Status checks
        if m.status != MandateStatus.AUTHORIZED:
            _blocked("MANDATE_NOT_ACTIVE", f"Mandate is {m.status.value}, cannot debit.")
        if time.time() > m.expires_at:
            m.status = MandateStatus.EXPIRED
            _blocked("MANDATE_EXPIRED", "Mandate validity window has passed.")

        # 3. Guardrail: per-transaction cap (protects against a single
        #    hallucinated/oversized cart wiping the whole mandate at once)
        if amount_paise > m.per_txn_cap_paise:
            _blocked("PER_TXN_CAP_EXCEEDED",
                      f"Rs {amount_paise/100:.2f} exceeds per-transaction cap of Rs {m.per_txn_cap_paise/100:.2f}.")

        # 4. Guardrail: cannot exceed remaining blocked balance
        if amount_paise > m.available_paise:
            _blocked("INSUFFICIENT_MANDATE_BALANCE",
                      f"Only Rs {m.available_paise/100:.2f} left on this mandate.")

        # 5. All checks passed -> commit
        m.utilised_paise += amount_paise
        entry = LedgerEntry(entry_id, amount_paise, reason, "SUCCESS", time.time(), idempotency_key)
        m.ledger.append(entry)
        m._seen_idempotency_keys.add(idempotency_key)
        if m.available_paise <= 0:
            m.status = MandateStatus.EXHAUSTED
        return entry

    def _require(self, mandate_id: str) -> Mandate:
        m = self._mandates.get(mandate_id)
        if not m:
            raise DebitError("MANDATE_NOT_FOUND", "No such mandate.")
        return m


store = MandateStore()
