# Agentic Checkout — Frictionless AI Commerce via UPI Reserve Pay

An AI shopping agent that authorizes a spending mandate once, then searches,
carts, and checks out on the user's behalf with **zero further PIN prompts**
— built on the pattern behind Razorpay's UPI Reserve Pay (Single Block,
Multi Debit / SBMD).

## What's real vs. simulated (read this first)

Reserve Pay's multi-debit (SBMD) capability requires Razorpay to activate it
on a **business account**, which isn't self-serve and isn't available to a
student project on a deadline. So this project is honest about the split:

| Layer | Status |
|---|---|
| Order creation, Razorpay Checkout.js, HMAC signature verification | **Real** — genuine Razorpay Test Mode API calls (`backend/razorpay_client.py`) |
| Initial "block funds" authorization | **Real** Razorpay test-mode payment, via Checkout.js |
| Multiple zero-click debits against the blocked amount | **Simulated** in `backend/mandate.py`, using the same state machine (`INITIALIZED → AUTHORIZED → charge`) and guardrails a real SBMD integration needs |
| Product catalog | Mocked, in-memory |

This split is deliberate and is the most defensible way to present this in
an interview: the parts that *can* be real, are; the part that's gated
behind business approval is clearly labeled as a simulation of the same
lifecycle Razorpay's own docs describe.

## Architecture

```
                    ┌─────────────────────┐
   User message ──▶ │   LangGraph agent    │
                    │  (backend/agent.py)  │
                    └──────────┬───────────┘
                               │ proposes an action
                               │ (search / add_to_cart / checkout)
                               ▼
                    ┌─────────────────────┐
                    │   Mandate Engine     │   ◀── this is the "AI Risk
                    │  (backend/mandate.py)│       Manager" — deterministic,
                    │  idempotency, caps,  │       not LLM-controlled
                    │  expiry, ledger      │
                    └──────────┬───────────┘
                               │ only if all guardrails pass
                               ▼
                         money moves
```

The core design decision: **the LLM never touches money directly.** It can
only propose a checkout. Every proposal is re-validated in plain
deterministic code against the mandate's guardrails before anything is
debited. This separation is what you should lead with when asked "how do
you stop the agent from doing something dangerous."

### Guardrails implemented in the mandate engine
- **Idempotency keys** — an agent retry after a network timeout can't cause
  a double debit.
- **Per-transaction cap** — independent of the total mandate size, caps any
  single AI-initiated purchase (protects against one oversized/hallucinated
  cart wiping the mandate in one shot).
- **Remaining-balance check** — can never debit more than what's left.
- **Expiry** — mandate stops working after its validity window (capped at
  Razorpay's real 90-day SBMD limit).
- **Fail-closed** — any unexpected state blocks the debit rather than
  allowing it.
- **Full audit ledger** — every attempt (successful or blocked) is recorded
  with a reason.

## Project layout

```
agentic-checkout/
  backend/
    main.py              FastAPI app — chat, mandate, product endpoints
    agent.py             LangGraph state graph (search/cart/checkout nodes)
    mandate.py           Guardrailed mandate + ledger engine
    razorpay_client.py   Real Razorpay Orders API + signature verification
    inventory.py         Mock product catalog
    requirements.txt
    .env.example
  frontend/
    index.html           Single-file UI: chat + reserve meter + ledger
  README.md
  CHALLENGES.md          Interview prep: what broke, how it was fixed
```

## Running it

### 1. Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# edit .env: paste your Razorpay TEST MODE key id + secret
# (Dashboard -> Settings -> API Keys, with "Test Mode" toggled on)
uvicorn main:app --reload --port 8000
```

### 2. Frontend
Just open `frontend/index.html` in a browser. It talks to
`http://localhost:8000` by default (editable in the top bar). If the
backend isn't reachable, it automatically falls back to **local demo
mode** — a client-side mirror of the same guardrail logic — so the UI is
still fully demoable with no setup.

### 3. Try it
1. Set a block amount (e.g. ₹5000) and per-transaction cap (e.g. ₹2000),
   click **Authorize with Razorpay (test)**.
2. Complete the Razorpay test checkout (use a
   [test card or test UPI VPA](https://razorpay.com/docs/payments/payments/test-card-upi-details/)).
3. Ask the agent things like:
   - `show me earbuds under 3000`
   - `add sku_001 to cart`
   - `checkout`
4. Watch the reserve meter deplete and the ledger fill in as the agent
   spends — with zero further authentication.
5. Try to break it: add several items that exceed your per-transaction cap
   and checkout — the agent will refuse and tell you why.

## Why this maps to Razorpay's actual tracks

- **AI Risk Manager** — the guardrail/ledger engine is the centerpiece.
- **AI Revenue Recovery / Agentic Commerce** — zero-click repeat purchases
  are exactly what Reserve Pay is positioned for (Razorpay's own materials
  frame it as an execution layer for agent-led commerce).
- Honest about scope — doesn't claim SBMD business activation it doesn't have.
