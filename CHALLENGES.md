# Challenges & how they were addressed

Use this as prep notes for "what broke and how did you fix it" — the
question the placement post says to expect.

### 1. LLMs are non-deterministic; money can't be
An LLM occasionally mis-parses intent or hallucinates a SKU/quantity. If the
agent could call a "debit" function directly, that non-determinism becomes
a financial bug. **Fix:** hard separation between the agent (proposes an
action) and the mandate engine (a plain deterministic function that
validates and either commits or rejects). The LLM's only power is to
*suggest*; a rule-based layer has final say. This is why `mandate.py` has
no dependency on `agent.py` at all — it can be unit-tested completely
independently of anything AI-related.

### 2. Double-debits from agent retries
Agent frameworks retry on timeouts/errors. If a debit request is retried
after actually succeeding server-side (but the response was lost), a naive
implementation double-charges. **Fix:** every debit call carries a caller-
generated idempotency key; the mandate engine stores seen keys and returns
the original result for a repeat key instead of processing it again.

### 3. One oversized cart draining the whole mandate
A total-mandate cap alone doesn't stop a single large (possibly
hallucinated) cart from consuming the entire reserve in one purchase.
**Fix:** added an independent per-transaction cap, set by the user at
authorization time, that every single debit is checked against —
regardless of how much total headroom remains.

### 4. Trusting the frontend's "payment succeeded" callback
Razorpay Checkout.js calls a JS handler on success, but that callback comes
from the browser and is trivially fake-able (e.g. via devtools). **Fix:**
the backend never trusts the callback directly — it re-derives the HMAC-
SHA256 signature from the order id + payment id using the secret key
server-side and compares it with `hmac.compare_digest` before flipping the
mandate to `AUTHORIZED`.

### 5. Real SBMD activation isn't available to a demo/student account
Initial plan assumed live multi-debit Reserve Pay. Razorpay's docs make
clear this requires support-gated business activation. **Fix:** rather than
fake it silently, split the system explicitly — real Orders API +
Checkout.js for authorization, simulated (but architecturally faithful)
ledger for the repeat debits — and documented the split up front. Chose
transparency over a demo that looks real but can't survive a technical
follow-up question.

### 6. Demo reliability during the actual interview
A live backend dependency is a single point of failure during a demo (wifi,
forgotten `.env`, etc). **Fix:** the frontend pings `/api/health` on load
and silently falls back to a client-side "local demo mode" that mirrors the
same guardrail logic in JS, so the UI/UX can always be shown even if the
Python backend isn't running.

### Open problems / what I'd do next
- Persist mandates in Redis/Postgres instead of an in-memory dict (current
  version resets on server restart).
- Rate-limit debits per unit time, not just per-transaction/total, to catch
  runaway agent loops.
- Real webhook handling (`payment.captured`, `refund.processed`) instead of
  only the synchronous verify path.
