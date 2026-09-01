"""
LangGraph shopping agent.

Graph shape:

    interpret --> search        |
              --> cart_update   +--> compose_reply --> END
              --> checkout      |   (preview only, does NOT debit)
              --> chit_chat     |

Two separate LLM touchpoints:

  1. `interpret`      -- message -> structured intent+slots.
  2. `compose_reply`  -- structured FACTS -> natural language. Cannot see
                         or influence the action; the action already
                         happened (or was previewed) by the time this runs.

Both use Gemini when GEMINI_API_KEY is set, and fall back to deterministic
templates/rules otherwise.

CHECKOUT IS NOW TWO SEPARATE STEPS, on purpose:

  1. `node_checkout` (in the graph, triggered by "checkout"/"buy it" style
     messages) only PREVIEWS the order -- it computes the total and runs a
     non-committing guardrail check, but never calls mandate.store.debit().
     It produces `facts = {"type": "confirm_checkout", ...}`, and the
     frontend renders this as a "Confirm & Pay" button in the chat. Nothing
     is spent yet.
  2. `execute_checkout()` is a plain function (NOT a graph node, not wired
     to any LLM at all) that main.py calls only when the user explicitly
     clicks that button, via POST /api/checkout/confirm. This is the only
     code path in the whole project that can call mandate.store.debit().

This split exists because a purely conversational "say checkout and it's
done" flow is bad UX and bad safety posture for a payments demo: the user
should see the exact amount and explicitly confirm, the same way a normal
checkout button works, even though the *whole point* of the mandate is
that no further PIN/OTP is needed once they do confirm.
"""
from __future__ import annotations
import contextvars
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

import acp
import inventory
import mandate as mandate_mod
import merchant as merchant_mod

logger = logging.getLogger("agent")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Configurable via env so a future model bump/deprecation doesn't require a
# code change -- Google ships new Gemini versions frequently (e.g. Gemini
# 2.0 was shut down in June 2026). gemini-2.5-flash is the current stable,
# well-established choice as of Aug 2026.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Per-request Gemini key override, set by main.py when a session supplied
# its own key via the "no .env keys -> ask in browser" popup. A ContextVar
# (not a plain global) so concurrent requests from different sessions never
# leak each other's keys -- it's request-scoped, propagates correctly
# through FastAPI's threadpool, and always resets itself even on error.
# When unset (the common case: server-side GEMINI_API_KEY from .env),
# behavior is identical to before this existed.
_gemini_key_ctx: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "gemini_key_override", default=None
)


def _effective_gemini_key() -> str:
    return _gemini_key_ctx.get() or GEMINI_API_KEY


class gemini_key_override:
    """
    Context manager: `with agent.gemini_key_override(session_key): ...`
    temporarily makes every _call_gemini() inside the block use
    `session_key` instead of the server's GEMINI_API_KEY. Falls back to
    GEMINI_API_KEY automatically if session_key is None/empty -- so it's
    always safe to call unconditionally with whatever the session has on
    file, even "nothing".
    """

    def __init__(self, key: Optional[str]):
        self.key = key
        self._token = None

    def __enter__(self):
        self._token = _gemini_key_ctx.set(self.key)
        return self

    def __exit__(self, exc_type, exc, tb):
        _gemini_key_ctx.reset(self._token)
        return False


class AgentUnavailableError(RuntimeError):
    """
    Raised whenever Gemini is required but unconfigured or erroring, so
    callers get a clean, honest failure instead of the agent silently
    degrading to keyword-matching and template sentences that only look
    intelligent. main.py catches this and returns a clear HTTP error
    (or, for the one place money may already have moved -- see
    _checkout_system_notice below -- a factual, clearly-labeled notice
    instead of a fabricated "AI" reply).
    """
    pass

# Each chat message costs up to 2 Gemini calls (interpret + compose_reply),
# so a 20/day free-tier key covers roughly 10 messages before every
# request starts failing with 429 RESOURCE_EXHAUSTED. Once that happens,
# retrying on every single message is pure waste: it adds a network
# round-trip's worth of latency to a request that's guaranteed to fail,
# and floods the logs. This cooldown makes _call_gemini fail FAST and
# LOCAL once a 429 is seen, until Google's own suggested retryDelay (or a
# 60s default if that's not present in the error) has elapsed -- at which
# point it starts trying Gemini again automatically. No behavior change
# otherwise; the existing rule_based/template fallbacks still fire either way.
_gemini_cooldown_until = 0.0
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)")


class AgentState(TypedDict, total=False):
    user_message: str
    mandate_id: Optional[str]
    cart: List[Dict]
    last_shown_items: List[Dict]     # items from the most recent search, for "buy it" style follow-ups
    intent: str
    slots: Dict
    facts: Dict                       # structured, ground-truth outcome of the action node
    reply: str                         # final natural-language reply (LLM-composed or fallback)
    action_result: Optional[Dict]      # same as facts, kept for the frontend


# --------------------------------------------------------------------------
# Node 1: interpret user message -> {intent, slots}
# --------------------------------------------------------------------------

INTERPRET_SYSTEM_PROMPT = """You are the NLU layer of a shopping agent. Read the user's message and the
current cart, then output STRICT JSON only (no prose, no markdown fences) with this shape:

{"intent": "search" | "add_to_cart" | "remove_from_cart" | "view_cart" | "checkout" | "chit_chat",
 "query": "<search text, or the product name/description the user wants to add, else empty string>",
 "max_price": <number or null>,
 "sku_id": "<sku id ONLY if the user typed an exact sku id, else empty string>"}

Rules:
- "checkout" only when the user clearly wants to pay / place the order now ("checkout", "buy it", "pay now").
- If the user names or describes a product they want to buy/add (e.g. "buy the usb c charger",
  "add the yoga mat"), that's "add_to_cart" with "query" set to their description of the item --
  a separate deterministic step resolves that description against the real catalog, so just pass
  their words through in "query". Do NOT invent a sku_id for this.
- If the user names a product type without any buy/add intent (e.g. "show me earbuds"), that's "search".
- Only set "sku_id" if the user's message literally contains a string like "sku_001".
"""


def node_interpret(state: AgentState) -> AgentState:
    message = state["user_message"]
    cart = state.get("cart", [])
    if not _effective_gemini_key():
        raise AgentUnavailableError(
            "No Gemini API key is configured for this session. Add one in the "
            "API keys prompt, or set GEMINI_API_KEY in .env on the server."
        )
    try:
        parsed = _gemini_interpret(message, cart)
    except Exception as e:
        raise AgentUnavailableError(f"Gemini is currently unavailable: {e}") from e
    state["intent"] = parsed.get("intent", "chit_chat")
    state["slots"] = parsed
    return state


def _call_gemini(system: str, user_content: str, max_tokens: int = 300) -> str:
    global _gemini_cooldown_until

    remaining = _gemini_cooldown_until - time.time()
    if remaining > 0:
        raise RuntimeError(f"Gemini in rate-limit cooldown for {remaining:.0f}s more, skipping network call.")

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_effective_gemini_key())
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or " 429" in msg:
            m = _RETRY_DELAY_RE.search(msg)
            delay = float(m.group(1)) + 2 if m else 60.0  # +2s safety margin over Google's own suggested delay
            _gemini_cooldown_until = time.time() + delay
            logger.warning("Gemini rate-limited; pausing Gemini calls for %.0fs", delay)
        raise

    text = (resp.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    return text


def _gemini_interpret(message: str, cart: List[Dict]) -> Dict:
    text = _call_gemini(INTERPRET_SYSTEM_PROMPT, f"Cart: {json.dumps(cart)}\nMessage: {message}")
    text = re.sub(r"^```json|```$", "", text).strip()
    return json.loads(text)  # let caller catch + raise AgentUnavailableError


# --------------------------------------------------------------------------
# Action nodes -- 100% deterministic. These decide WHAT happened.
# They never touch the LLM and never write prose; they only write `facts`.
# --------------------------------------------------------------------------

def node_search(state: AgentState) -> AgentState:
    slots = state["slots"]
    results = inventory.search(slots.get("query", ""), max_price=slots.get("max_price"))
    shown = results[:5]
    state["facts"] = {"type": "search_results", "query": slots.get("query", ""), "items": shown}
    state["last_shown_items"] = shown  # so a later "buy it" can resolve back to these
    return state


# Words that carry no product information -- stripped before fuzzy-matching
# a phrase like "yes buy it" or "add that one" against the catalog, so we
# don't accidentally match "it"/"that"/"one" against some item's tags.
_ITEM_QUERY_STOPWORDS = {
    "buy", "it", "that", "this", "please", "yes", "get", "add", "the",
    "one", "item", "now", "to", "cart", "me", "a", "an", "for",
}


def _extract_item_query(text: str) -> str:
    tokens = [w for w in re.findall(r"[a-zA-Z0-9\-]+", text.lower()) if w not in _ITEM_QUERY_STOPWORDS]
    return " ".join(tokens)


def node_cart_update(state: AgentState) -> AgentState:
    slots = state["slots"]
    cart = state.get("cart", [])
    intent = state["intent"]
    facts: Dict = {"type": intent}

    if intent == "add_to_cart":
        item = None

        # 1. Explicit sku id (typed or already resolved by interpret).
        sku_id = slots.get("sku_id") or ""
        match = re.search(r"sku_\d+", state["user_message"].lower())
        if not sku_id and match:
            sku_id = match.group(0)
        if sku_id:
            item = inventory.get_by_id(sku_id)

        # 2. No sku, but the message names/describes a product -- resolve it
        #    deterministically against the real catalog (no LLM involved in
        #    the decision of which item this is; inventory.search is a
        #    plain keyword match). E.g. "buy the usb c fast charger".
        if not item:
            cleaned = _extract_item_query(slots.get("query") or state["user_message"])
            if cleaned:
                candidates = inventory.search(cleaned)
                if candidates:
                    # Prefer whichever candidate actually contains the most
                    # of the query's meaningful words, rather than just
                    # taking inventory.search's first result -- avoids
                    # picking an unrelated item that happened to match one
                    # weak token.
                    tokens = [t for t in cleaned.split() if len(t) >= 3]
                    def _score(candidate):
                        hay = " ".join([candidate["name"].lower(), candidate["category"].lower(), *candidate.get("tags", [])])
                        return sum(1 for t in tokens if t in hay)
                    item = max(candidates, key=_score) if tokens else candidates[0]

        # 3. Still nothing -- maybe they're referring back to what was just
        #    shown ("yes buy it") without naming it. If exactly one item was
        #    last shown, that's unambiguous. If several were shown, ask
        #    which one instead of guessing.
        if not item:
            last_shown = state.get("last_shown_items") or []
            if len(last_shown) == 1:
                item = last_shown[0]
            elif len(last_shown) > 1:
                facts.update({
                    "ok": False,
                    "reason": "ambiguous_reference",
                    "options": [i["name"] for i in last_shown],
                })
                state["facts"] = facts
                return state

        if not item:
            facts.update({"ok": False, "reason": "unknown_item"})
        else:
            cart.append(item)
            state["cart"] = cart
            facts.update({"ok": True, "added_item": item, "cart": cart, "cart_total": sum(i["price"] for i in cart)})

    elif intent == "remove_from_cart":
        match = re.search(r"sku_\d+", state["user_message"].lower())
        if match and cart:
            removed = next((i for i in cart if i["id"] == match.group(0)), None)
            cart = [i for i in cart if i["id"] != match.group(0)]
            state["cart"] = cart
            facts.update({"ok": True, "removed_item": removed, "cart": cart, "cart_total": sum(i["price"] for i in cart)})
        else:
            facts.update({"ok": False, "reason": "no_sku_given_or_empty_cart"})

    elif intent == "view_cart":
        facts.update({"ok": True, "cart": cart, "cart_total": sum(i["price"] for i in cart)})

    state["facts"] = facts
    return state


def node_checkout(state: AgentState) -> AgentState:
    """
    PREVIEW ONLY. This node never calls mandate.store.debit() -- see the
    module docstring for why checkout is deliberately split into a preview
    (this node) and a separate execute step (execute_checkout(), below,
    only reachable via an explicit user click in the frontend).

    It still runs a real (non-committing) guardrail check against the
    mandate so the "Confirm & Pay" button isn't shown for an order that's
    guaranteed to fail -- e.g. if the cart already exceeds the per-purchase
    cap, we say so up front instead of letting the user tap confirm and
    then get a rejection.
    """
    cart = state.get("cart", [])
    mandate_id = state.get("mandate_id")

    # "yes buy it" / "buy it" said right after a search, with nothing
    # actually in the cart yet: resolve against what was just shown rather
    # than failing with a bare "cart is empty". This was the exact bug from
    # testing -- shown two items, said "yes buy it", got an empty-cart
    # error instead of either buying the one clear match or being asked
    # which one was meant.
    if not cart:
        last_shown = state.get("last_shown_items") or []
        if len(last_shown) == 1:
            cart = [last_shown[0]]
            state["cart"] = cart
        elif len(last_shown) > 1:
            state["facts"] = {
                "type": "add_to_cart",
                "ok": False,
                "reason": "ambiguous_reference",
                "options": [i["name"] for i in last_shown],
            }
            return state
        else:
            state["facts"] = {"type": "confirm_checkout", "ok": False, "reason": "empty_cart"}
            return state

    if not mandate_id:
        state["facts"] = {"type": "confirm_checkout", "ok": False, "reason": "no_mandate"}
        return state

    total = sum(i["price"] for i in cart)
    m = mandate_mod.store.get(mandate_id)
    preflight_ok, preflight_reason = True, None
    if not m or m.status != mandate_mod.MandateStatus.AUTHORIZED:
        preflight_ok, preflight_reason = False, "your mandate isn't active"
    elif total > m.per_txn_cap_paise / 100:
        preflight_ok, preflight_reason = False, f"this exceeds your per-purchase cap of Rs {m.per_txn_cap_paise/100:.0f}"
    elif total > m.available_paise / 100:
        preflight_ok, preflight_reason = False, f"only Rs {m.available_paise/100:.0f} is left on your mandate"

    state["facts"] = {
        "type": "confirm_checkout",
        "ok": preflight_ok,
        "reason": preflight_reason,
        "items": [i["name"] for i in cart],
        "amount": total,
    }
    return state


def execute_checkout(cart: List[Dict], mandate_id: Optional[str]) -> Dict:
    """
    The ONLY function in this codebase that can move money. Called
    exclusively from POST /api/checkout/confirm, i.e. only after an
    explicit user click on the "Confirm & Pay" button -- never from the
    chat graph, never triggered by anything an LLM decided on its own.

    Execution itself goes through the Agentic Commerce Protocol (ACP)
    checkout-session lifecycle (see acp.py): a session is created from
    the cart and immediately completed with a payment_data token pointing
    at the user's Reserve Pay mandate. This is the same ACP surface
    exposed externally as real REST routes in main.py (POST
    /checkout_sessions, .../complete, etc.) -- the chat-driven confirm
    button and an external ACP-speaking agent both end up going through
    acp.complete_session(), which is the only function allowed to call
    mandate.store.debit().
    """
    if not cart:
        return {"type": "checkout", "ok": False, "reason": "empty_cart"}
    if not mandate_id:
        return {"type": "checkout", "ok": False, "reason": "no_mandate"}

    total = sum(i["price"] for i in cart)
    try:
        session = acp.create_session(cart, mandate_id=mandate_id)
        session = acp.complete_session(
            session.id,
            payment_data={"token": mandate_id, "provider": "reserve_pay_mandate"},
            idempotency_key=f"checkout_{uuid.uuid4().hex}",
        )
        return {
            "type": "checkout", "ok": True,
            "txn_id": session.order["id"], "amount": total,
            "items": [i["name"] for i in cart],
            "acp_checkout_session_id": session.id,
            "merchant": merchant_mod.MERCHANT["storefront_name"],
        }
    except acp.ACPError as e:
        return {"type": "checkout", "ok": False, "reason": e.code, "detail": e.message, "amount": total}


def node_chit_chat(state: AgentState) -> AgentState:
    state["facts"] = {"type": "chit_chat"}
    return state


# --------------------------------------------------------------------------
# compose_reply -- the ONLY thing allowed to write user-facing prose.
# Takes `facts` (ground truth, already decided) and phrases a reply.
# --------------------------------------------------------------------------

COMPOSE_SYSTEM_PROMPT = """You are the voice of a shopping agent for an AI-native checkout demo.
You will be given a JSON object of FACTS describing something that already happened or is being
previewed (a search, a cart change, or a checkout preview/result).

Write a short, natural, friendly reply (2-4 sentences max) describing these facts to the user.

Hard rules:
- CURRENCY: every amount in the JSON is already a plain number of Indian Rupees (not paise, not
  cents, not dollars). Always write it as "Rs <number>" using the number EXACTLY as given --
  never divide, multiply, add decimal points, or use a $ sign. For example if the JSON says
  899, write "Rs 899", never "$8.99" or "Rs 8.99".
- Never invent facts, prices, or item names not present in the JSON.
- If facts.type is "confirm_checkout": this is a PREVIEW, nothing has been charged yet.
  If facts.ok is true, state the total and tell the user to tap "Confirm & Pay" to complete it --
  never say the purchase is done. If facts.ok is false, explain facts.reason plainly.
- If facts.type is "checkout": this already happened. If facts.ok is true, confirm the amount was
  charged and that no PIN/OTP was needed since it was within the pre-authorized mandate. If false,
  clearly explain why it was blocked (facts.reason / facts.detail) -- do not soften or hide it.
- If facts.reason is "ambiguous_reference", ask the user which of facts.options they meant --
  do not guess which one.
- For search_results, list item names with prices; if items is empty, say nothing matched.
- Keep it concise. No markdown headers, no emoji spam.
- Never add your own sign-off, P.S., or "powered by" line -- that's appended separately.
"""


def _checkout_system_notice(f: Dict) -> str:
    """
    A minimal, factual, clearly-labeled notice used ONLY as the reply text
    when Gemini is unavailable AND facts.type == "checkout" -- i.e. only
    after execute_checkout() has already run and money may already have
    moved. This is deliberately not a general-purpose "personality"
    fallback (that's been removed): it's a safety-critical exception,
    because a completed debit's outcome can never be silently dropped just
    because the reply-writer is down. See compose_reply_from_facts below.
    """
    if f.get("ok"):
        merchant_bit = f" with {f['merchant']}" if f.get("merchant") else ""
        return f"[System notice] Checkout SUCCEEDED{merchant_bit}. Charged Rs {f['amount']}. Order {f['txn_id']}."
    reason = f.get("reason")
    if reason == "empty_cart":
        return "[System notice] Checkout not attempted: cart was empty."
    if reason == "no_mandate":
        return "[System notice] Checkout not attempted: no authorized mandate."
    return f"[System notice] Checkout FAILED: {f.get('detail', reason)}"


def compose_reply_from_facts(facts: Dict) -> str:
    if not _effective_gemini_key():
        if facts.get("type") == "checkout":
            return _checkout_system_notice(facts) + " (AI reply unavailable: no Gemini API key configured.)"
        raise AgentUnavailableError(
            "No Gemini API key is configured for this session. Add one in the "
            "API keys prompt, or set GEMINI_API_KEY in .env on the server."
        )
    try:
        reply = _call_gemini(COMPOSE_SYSTEM_PROMPT, json.dumps(facts), max_tokens=200)
        if not reply:
            raise ValueError("Gemini returned an empty reply")
    except Exception as e:
        if facts.get("type") == "checkout":
            return _checkout_system_notice(facts) + f" (AI reply unavailable: {e})"
        raise AgentUnavailableError(f"Gemini is currently unavailable: {e}") from e
    return _with_checkout_signature(reply, facts)


def _with_checkout_signature(reply: str, facts: Dict) -> str:
    """
    Appends a short P.S. after every SUCCESSFUL checkout. Done here in code
    rather than left to the LLM's system prompt so it's always exactly this
    line, worded identically, regardless of whether Gemini or the
    checkout-system-notice path produced the rest of the reply.
    """
    if facts.get("type") == "checkout" and facts.get("ok"):
        return f"{reply}\n\nP.S. — Powered by Agentic Checkout."
    return reply


def node_compose_reply(state: AgentState) -> AgentState:
    facts = state.get("facts", {"type": "chit_chat"})
    state["action_result"] = facts  # frontend still gets the raw structured facts too
    state["reply"] = compose_reply_from_facts(facts)
    return state


# --------------------------------------------------------------------------
# Graph wiring
# --------------------------------------------------------------------------

def route(state: AgentState) -> str:
    return {
        "search": "search",
        "add_to_cart": "cart_update",
        "remove_from_cart": "cart_update",
        "view_cart": "cart_update",
        "checkout": "checkout",
        "chit_chat": "chit_chat",
    }.get(state["intent"], "chit_chat")


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("interpret", node_interpret)
    g.add_node("search", node_search)
    g.add_node("cart_update", node_cart_update)
    g.add_node("checkout", node_checkout)
    g.add_node("chit_chat", node_chit_chat)
    g.add_node("compose_reply", node_compose_reply)

    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route, {
        "search": "search",
        "cart_update": "cart_update",
        "checkout": "checkout",
        "chit_chat": "chit_chat",
    })
    g.add_edge("search", "compose_reply")
    g.add_edge("cart_update", "compose_reply")
    g.add_edge("checkout", "compose_reply")
    g.add_edge("chit_chat", "compose_reply")
    g.add_edge("compose_reply", END)
    return g.compile()


graph = build_graph()


def run_agent(user_message: str, cart: List[Dict], mandate_id: Optional[str],
              last_shown_items: Optional[List[Dict]] = None) -> AgentState:
    initial: AgentState = {
        "user_message": user_message,
        "cart": cart,
        "mandate_id": mandate_id,
        "last_shown_items": last_shown_items or [],
    }
    return graph.invoke(initial)
