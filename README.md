# Agentic Checkout
### Frictionless AI Commerce, modeled on Razorpay UPI Reserve Pay

<img width="1859" height="632" alt="image" src="https://github.com/user-attachments/assets/5cd39de0-42af-4518-964a-c70acb06e282" />
<img width="1919" height="944" alt="image" src="https://github.com/user-attachments/assets/d6f6e8da-087d-4cb1-9b71-8c41281c6453" />

An AI shopping agent that authorizes a spending mandate **once**, then
searches, carts, and checks out on the user's behalf entirely in chat —
with **zero further PIN prompts, zero re-authentication, zero handoff to a
human**. The mandate engine mirrors the exact lifecycle of Razorpay's UPI
Reserve Pay (Single Block, Multi Debit / SBMD) — real Razorpay test-mode
Checkout.js handles the initial fund block, while the repeat zero-click
debits are simulated (see
[What's real vs. simulated](#whats-real-vs-simulated-read-this-first) below).

> **The idea in one line:** checkout shouldn't be the moment an AI shopping
> experience hands back to a human. This project shows what it feels like
> when it doesn't.

---

## Table of contents
- [Why this exists](#why-this-exists)
- [What's real vs. simulated](#whats-real-vs-simulated-read-this-first)
- [Architecture](#architecture)
- [Agentic Commerce Protocol (ACP)](#agentic-commerce-protocol-acp)
- [Guardrails](#guardrails-in-the-mandate-engine)
- [Project layout](#project-layout)
- [Quickstart](#quickstart)
- [Try it](#try-it)
- [Tech stack](#tech-stack)
- [Challenges & how they were solved](#challenges--how-they-were-solved)
- [Why this fits the Agentic Commerce track](#why-this-fits-the-agentic-commerce-track)

---

## Why this exists

Every extra step in checkout is a place a customer leaves. Conversational
shopping agents today can recommend and cart just fine — but the moment it's
time to actually pay, most of them still have to hand off to a human, because
repeated authentication doesn't work inside a chat flow.

Razorpay's UPI Reserve Pay is built to remove that wall: block funds once
with **Single Block, Multi Debit (SBMD)**, and an approved agent can debit
against that block repeatedly, with no further authentication. That's the
missing piece for agent-led commerce to close the loop — from *"I want
this"* to *"paid"* — inside a single conversation.

**Agentic Checkout** is a working demonstration of that loop end to end.
The initial fund block runs on real Razorpay test-mode Checkout.js; the
repeat zero-click debits run on a mandate engine built to the same
lifecycle, since real SBMD access requires business-account activation this
project doesn't have (details below).

```
"show me earbuds under 3000"  →  agent searches, shows options
"add sku_001 to cart"         →  agent carts it
"checkout"                    →  agent renders a Confirm & Pay preview
[tap Confirm & Pay]           →  paid. no PIN, no OTP, no redirect.
"add sku_002, checkout"       →  repeat purchase, same conversation,
                                  still zero further auth
```

## What's real vs. simulated (read this first)

Reserve Pay's multi-debit (SBMD) capability requires Razorpay to activate it
on a **business account**, which isn't self-serve and isn't available to a
student project on a deadline. So this project is upfront about the split —
the parts that *can* be real, are:

| Layer | Status |
|---|---|
| Order creation, Razorpay Checkout.js, HMAC signature verification | **Real** — genuine Razorpay Test Mode API calls (`backend/razorpay_client.py`) |
| Initial "block funds" authorization | **Real** Razorpay test-mode payment, via Checkout.js |
| Multiple zero-click debits against the blocked amount | **Simulated** in `backend/mandate.py`, using the same state machine (`INITIALIZED → AUTHORIZED → charge`) and guardrails a real SBMD integration needs |
| Checkout **execution/transaction layer** | **Real protocol shape** — implements the [Agentic Commerce Protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol) checkout-session lifecycle (`backend/acp.py`), exposed as real `POST /checkout_sessions` REST routes in `main.py` |
| Payment handler behind ACP | **Seller-backed payment handler** pattern (explicitly supported by ACP), backed by `mandate.py` — not Stripe's Shared Payment Token network, which requires a real onboarded merchant |
| Merchant identity (business name, MCC, support contact) | **Simulated** — fictional storefront details in `backend/merchant.py` |
| Product catalog | Mocked, in-memory |

This is the most defensible way to present it in an interview: nothing here
pretends to be more real than it is, and the part that's gated behind
business approval is clearly labeled as a simulation of the exact lifecycle
Razorpay's own docs describe.

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
                    │   Mandate Engine     │  deterministic, not
                    │  (backend/mandate.py)│  LLM-controlled —
                    │  idempotency, caps,  │  guardrails + ledger
                    │  expiry, ledger      │
                    └──────────┬───────────┘
                               │ only if all guardrails pass
                               ▼
                         money moves
```

The core design decision: **the LLM never touches money directly.** It can
only *propose* a checkout. Every proposal is re-validated in plain,
deterministic code against the mandate's guardrails before anything is
debited — which is what makes "let the agent just handle it" something a
business could realistically turn on.

Checkout is split into two steps for the same reason: saying "checkout" in
chat only produces a non-committing **preview** — item list, total, a live
guardrail check — rendered as an actual **"Confirm & Pay" card in the chat
UI** (not just text). Nothing is charged until the user explicitly taps that
button, which hits a separate endpoint (`POST /api/checkout/confirm`) — the
only code path in the project allowed to reach the mandate engine's debit
function. While the agent is generating a reply (or the demo-mode stand-in
is "thinking"), the chat shows an animated typing indicator instead of
appearing to hang.

### Agentic Commerce Protocol (ACP)

Execution — the actual "move money" step — goes through the
[Agentic Commerce Protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol),
the open standard co-developed by OpenAI and Stripe for how an agent, buyer,
and merchant complete a purchase. `backend/acp.py` implements the ACP
**checkout-session** resource and its lifecycle, exposed by `main.py` as
real REST routes:

| Route | Purpose |
|---|---|
| `POST /checkout_sessions` | Create a session from cart items |
| `POST /checkout_sessions/{id}` | Update buyer / fulfillment address |
| `GET /checkout_sessions/{id}` | Retrieve current session state |
| `POST /checkout_sessions/{id}/complete` | Pay → creates an order (only entry point to `mandate.store.debit()`) |
| `POST /checkout_sessions/{id}/cancel` | Cancel a session |

Both the chat-driven "Confirm & Pay" button (`agent.execute_checkout`) and
any external ACP-speaking agent go through the **exact same**
`acp.complete_session()` call — there's no separate, less-guarded path for
the in-app chat. The merchant (see `backend/merchant.py`) declares a single
accepted payment handler, `reserve_pay_mandate`, backed by the mandate
engine above.

## Guardrails in the mandate engine

Removing every friction step only helps growth if it doesn't also remove
safety. Every debit — no matter which path triggered it — passes through:

- **Idempotency keys** — an agent retry after a network timeout can't cause a double debit.
- **Per-transaction cap** — independent of total mandate size, so one oversized or hallucinated cart can't drain the whole mandate in one purchase.
- **Remaining-balance check** — can never debit more than what's left.
- **Expiry** — mandate stops working after its validity window (capped at Razorpay's real 90-day SBMD limit).
- **Fail-closed** — any unexpected state blocks the debit rather than allowing it.
- **Full audit ledger** — every attempt, successful or blocked, is recorded with a reason.

## Project layout

```
agentic-checkout/
  backend/
    main.py              FastAPI app — chat, mandate, ACP, product endpoints
    agent.py             LangGraph state graph (search/cart/checkout nodes)
    acp.py               Agentic Commerce Protocol checkout-session engine
    merchant.py          Dummy merchant profile used by ACP responses
    mandate.py           Guardrailed mandate + ledger engine
    razorpay_client.py   Real Razorpay Orders API + signature verification
    inventory.py         Mock product catalog
    requirements.txt
    .env.example
  frontend/
    index.html           Single-file UI: chat (Confirm & Pay card +
                          thinking indicator) + reserve meter + ledger
  README.md
  CHALLENGES.md          Interview prep: what broke, how it was fixed
```

## Quickstart

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
mode** — a client-side mirror of the same guardrail logic — so the UI stays
fully demoable with no setup.

### API keys: `.env`, in-browser prompt, or skip entirely
You don't need `.env` filled in to run this. On load, the frontend asks the
backend what's configured (`GET /api/config/status`):

- If the server already has `GEMINI_API_KEY` / `RAZORPAY_KEY_ID` +
  `RAZORPAY_KEY_SECRET` in `.env`, nothing else happens.
- If either is missing, a popup asks for that session's own keys instead
  (`POST /api/config/keys`). Keys are held **only in server memory**, per
  browser session, for the life of the process — never written to disk,
  never logged, never echoed back in full in any response.
- If you skip the popup (or leave a field blank), that session runs in
  **test mode**: Gemini calls are simply never attempted (rule-based NLU /
  template replies take over), and Razorpay calls use a fully local mock
  order plus a test-only authorization endpoint
  (`/api/mandate/authorize_test`) — no network call, no real payment rail
  touched. A diagonal **TEST MODE** ribbon stays pinned top-right the whole
  session as a reminder nothing is a real charge.

## Try it

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
5. Ask for a second item and checkout again — same block, same
   conversation, still zero-click.
6. Try to break it: add several items that exceed your per-transaction cap
   and checkout — the agent refuses and tells you why.

## Tech stack

| Layer | Choice |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Google Gemini (`google-genai`), with rule-based fallback when unconfigured |
| Backend | FastAPI + Pydantic + Uvicorn |
| Payments | Razorpay Orders API, Checkout.js, server-side HMAC-SHA256 signature verification |
| Checkout protocol | [Agentic Commerce Protocol](https://github.com/agentic-commerce-protocol/agentic-commerce-protocol) |
| Frontend | Single-file HTML/CSS/JS — no build step |

## Challenges & how they were solved

Full write-up in [`CHALLENGES.md`](./CHALLENGES.md). Highlights:

- **Env vars silently not loading** — `python-dotenv` was in
  `requirements.txt` but `load_dotenv()` was never called, so
  `os.environ.get()` in `agent.py` / `razorpay_client.py` always returned
  empty strings (both modules read env vars at *import time*). This caused
  two seemingly unrelated symptoms at once: Razorpay mandate creation
  500ing with "keys not set," and the agent silently falling back to
  rule-based templates instead of calling Gemini. Fixed by moving
  `load_dotenv()` to the top of `main.py`, before those modules are
  imported — verified via `TestClient` that `/api/health` reports
  `razorpay_configured: true`.
- **Frontend not loading** — the static file path was never registered in
  `main.py`, so `frontend/index.html` 404'd. Fixed by mounting it properly.
- **Currency hallucination** — once Gemini was wired in, it started
  quoting rupee prices as dollars (₹899 → "$8.99"). Fixed with an explicit
  system-prompt rule: all numbers are already in rupees, render as
  `Rs <number>` exactly as given, never convert.

## Why this fits the Agentic Commerce track

- **Zero-click repeat purchases** are exactly what Reserve Pay is
  positioned for — Razorpay's own materials frame it as an execution layer
  for agent-led commerce. This project models that lifecycle faithfully,
  even though live SBMD access is gated behind business-account activation.
- **Conversion, not just conversation** — the project's whole point is
  closing the loop from recommendation to paid order without a human
  handoff, which is the actual growth lever agentic commerce is chasing.
- **Honest about scope** — doesn't claim SBMD business activation it
  doesn't have, and says so clearly rather than papering over it.
