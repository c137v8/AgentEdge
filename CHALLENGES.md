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

### 6. Real bug from testing: LLM hallucinating currency
Once Gemini was wired into `compose_reply`, it started quoting prices like
"$8.99" for an item that costs Rs 899 -- it silently treated the plain
rupee integer as if it were cents and switched to dollars, because the
prompt never said not to. **Fix:** made the currency rule explicit and
concrete in the system prompt ("every number is already rupees, write it
as 'Rs <number>' exactly as given, never divide/multiply/add decimals/use
a $ sign"), with a worked example.

### 7. Real bug from testing: item names couldn't be added, only exact SKUs
The agent required an exact `sku_001`-style id to add anything to cart --
reasonable to prevent SKU hallucination, but "buy the usb c fast charger"
silently failed and fell back to a generic search instead. **Fix:** added a
deterministic (non-LLM) resolution step against the real catalog via
`inventory.search`, scored by how many meaningful query words match.
Found a second bug in the process: `inventory.search` did plain substring
matching, so a short token like "c" (from "usb c charger") matched as a
substring of "ele**c**tronics" and silently resolved to the wrong product.
Fixed by requiring matched tokens to be at least 3 characters.

### 8. UX gap: ambiguous "buy it" after multiple search results
Shown two items and told "yes buy it," the agent had no way to know which
one was meant. **Fix:** the backend tracks `last_shown_items` per session.
If exactly one item was last shown, "buy it" resolves to it unambiguously;
if several were shown, the agent asks which one instead of guessing.

### 9. Checkout was one step; now it's two, on purpose
Originally, saying "checkout" immediately called `mandate.store.debit()`.
**Fix:** split into `node_checkout` (non-committing preview -- computes the
total and previews the guardrail result without committing it) and
`execute_checkout()` (the only function that can call `store.debit()`,
reachable only via `POST /api/checkout/confirm`, which the frontend calls
exclusively from an explicit "Confirm & Pay" button tap in the chat). The
chat graph can propose a checkout; it can never execute one.

