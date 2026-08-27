"""
LangGraph shopping agent.

Graph shape:

    interpret --> search        |
              --> cart_update   +--> compose_reply --> END
              --> checkout      |
              --> chit_chat     |

Two separate LLM touchpoints, each with a narrow, specific job:

  1. `interpret`      -- reads the raw message, outputs structured intent+slots.
  2. `compose_reply`  -- reads structured FACTS about what just happened
                         (a search result list, a blocked debit and why, a
                         successful transaction id, etc.) and phrases a
                         natural reply. It cannot see or influence the
                         action -- the action already happened by the time
                         this node runs.

Both use Gemini (via the Google GenAI SDK) when GEMINI_API_KEY is set, and
fall back to small deterministic templates/rules when it isn't (so the demo
still runs end-to-end with no key / no internet, e.g. mid-interview wifi
issues).

The architectural point to say out loud in the interview: the LLM is
NEVER the thing that decides whether money moves, and it never writes to
the mandate ledger directly. `search`/`cart_update`/`checkout` are plain
deterministic Python functions that run first and produce a `facts` dict.
The LLM only turns already-decided facts into English *afterwards*. If the
LLM is unavailable, hallucinating, or compromised, the worst it can do is
phrase a message badly -- it cannot spend a rupee or misreport what
actually happened, because the facts dict is the single source of truth,
and the frontend also renders it verbatim (cart panel, ledger) regardless
of what the LLM says in the chat bubble.
"""
from __future__ import annotations
import json
import logging
import os
import re
import uuid
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

import inventory
import mandate as mandate_mod

logger = logging.getLogger("agent")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Configurable via env so a future model bump/deprecation doesn't require a
# code change -- Google ships new Gemini versions frequently (e.g. Gemini
# 2.0 was shut down in June 2026). gemini-2.5-flash is the current stable,
# well-established choice as of Aug 2026; gemini-3.7-flash is the newest
# frontier Flash model if you want to try it.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class AgentState(TypedDict, total=False):
    user_message: str
    mandate_id: Optional[str]
    cart: List[Dict]
    intent: str
    slots: Dict
    facts: Dict                     # structured, ground-truth outcome of the action node
    reply: str                       # final natural-language reply (LLM-composed or fallback)
    action_result: Optional[Dict]    # same as facts, kept for the frontend


# --------------------------------------------------------------------------
# Node 1: interpret user message -> {intent, slots}
# --------------------------------------------------------------------------

INTERPRET_SYSTEM_PROMPT = """You are the NLU layer of a shopping agent. Read the user's message and the
current cart, then output STRICT JSON only (no prose, no markdown fences) with this shape:

{"intent": "search" | "add_to_cart" | "remove_from_cart" | "view_cart" | "checkout" | "chit_chat",
 "query": "<search text if intent is search, else empty string>",
 "max_price": <number or null>,
 "sku_id": "<sku id if the user referenced a specific item they've already seen, else empty string>"}

Rules:
- "checkout" only when the user clearly wants to pay / place the order now.
- If the user names a product type without saying "buy now", that's "search".
- Never invent a sku_id that wasn't shown to them.
"""


def _rule_based_interpret(message: str) -> Dict:
    m = message.lower()
    has_sku = re.search(r"\bsku_\d+\b", m)

    # Order matters: "add sku_001 to cart" contains the word "cart", so a
    # naive check-for-"cart"-first would misclassify this as "view_cart".
    # Checking for an explicit add verb (or a referenced sku id) first
    # avoids that. Caught during testing -- see CHALLENGES.md.
    if any(w in m for w in ["checkout", "buy now", "place order", "pay now", "confirm order"]):
        return {"intent": "checkout", "query": "", "max_price": None, "sku_id": ""}
    if any(w in m for w in ["remove", "delete item", "take out"]):
        return {"intent": "remove_from_cart", "query": "", "max_price": None, "sku_id": ""}
    if any(w in m for w in ["add", "get me", "i want", "i'll take"]) or (has_sku and "buy" in m):
        price_match = re.search(r"under\s*(?:rs\.?|₹)?\s*(\d+)", m)
        return {
            "intent": "add_to_cart" if has_sku else "search",
            "query": message,
            "max_price": int(price_match.group(1)) if price_match else None,
            "sku_id": has_sku.group(0) if has_sku else "",
        }
    if any(w in m for w in ["cart", "what's in my basket", "show basket"]):
        return {"intent": "view_cart", "query": "", "max_price": None, "sku_id": ""}
    if any(w in m for w in ["hi", "hello", "hey", "thanks", "thank you"]):
        return {"intent": "chit_chat", "query": "", "max_price": None, "sku_id": ""}
    price_match = re.search(r"under\s*(?:rs\.?|₹)?\s*(\d+)", m)
    return {"intent": "search", "query": message, "max_price": int(price_match.group(1)) if price_match else None, "sku_id": ""}


def _call_gemini(system: str, user_content: str, max_tokens: int = 300) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    return text


def _gemini_interpret(message: str, cart: List[Dict]) -> Dict:
    text = _call_gemini(INTERPRET_SYSTEM_PROMPT, f"Cart: {json.dumps(cart)}\nMessage: {message}")
    text = re.sub(r"^```json|```$", "", text).strip()
    return json.loads(text)  # let caller catch + fall back


def node_interpret(state: AgentState) -> AgentState:
    message = state["user_message"]
    cart = state.get("cart", [])
    if GEMINI_API_KEY:
        try:
            parsed = _gemini_interpret(message, cart)
        except Exception as e:
            logger.warning("Gemini interpret failed, falling back to rules: %s", e)
            parsed = _rule_based_interpret(message)
    else:
        parsed = _rule_based_interpret(message)
    state["intent"] = parsed.get("intent", "chit_chat")
    state["slots"] = parsed
    return state


# --------------------------------------------------------------------------
# Action nodes -- 100% deterministic. These decide WHAT happened.
# They never touch the LLM and never write prose; they only write `facts`.
# --------------------------------------------------------------------------

def node_search(state: AgentState) -> AgentState:
    slots = state["slots"]
    results = inventory.search(slots.get("query", ""), max_price=slots.get("max_price"))
    state["facts"] = {"type": "search_results", "query": slots.get("query", ""), "items": results[:5]}
    return state


def node_cart_update(state: AgentState) -> AgentState:
    slots = state["slots"]
    cart = state.get("cart", [])
    intent = state["intent"]
    facts: Dict = {"type": intent}

    if intent == "add_to_cart":
        sku_id = slots.get("sku_id") or ""
        match = re.search(r"sku_\d+", state["user_message"].lower())
        if not sku_id and match:
            sku_id = match.group(0)
        item = inventory.get_by_id(sku_id)
        if not item:
            facts.update({"ok": False, "reason": "unknown_sku", "sku_id": sku_id})
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
    The guarded chokepoint. This is a plain function -- no LLM call. The
    LLM downstream (compose_reply) only ever describes what this function
    already, irreversibly decided.
    """
    cart = state.get("cart", [])
    mandate_id = state.get("mandate_id")

    if not cart:
        state["facts"] = {"type": "checkout", "ok": False, "reason": "empty_cart"}
        return state
    if not mandate_id:
        state["facts"] = {"type": "checkout", "ok": False, "reason": "no_mandate"}
        return state

    total = sum(i["price"] for i in cart)
    idempotency_key = f"checkout_{uuid.uuid4().hex}"
    try:
        entry = mandate_mod.store.debit(
            mandate_id=mandate_id,
            amount_rupees=total,
            reason=f"Order: {', '.join(i['name'] for i in cart)}",
            idempotency_key=idempotency_key,
        )
        state["facts"] = {
            "type": "checkout", "ok": True, "txn_id": entry.id, "amount": total,
            "items": [i["name"] for i in cart],
        }
        state["cart"] = []
    except mandate_mod.DebitError as e:
        state["facts"] = {"type": "checkout", "ok": False, "reason": e.code, "detail": e.message, "amount": total}
    return state


def node_chit_chat(state: AgentState) -> AgentState:
    state["facts"] = {"type": "chit_chat"}
    return state


# --------------------------------------------------------------------------
# Node: compose_reply -- the ONLY node allowed to write user-facing prose.
# Takes `facts` (ground truth, already decided) and phrases a reply.
# --------------------------------------------------------------------------

COMPOSE_SYSTEM_PROMPT = """You are the voice of a shopping agent for an AI-native checkout demo.
You will be given a JSON object of FACTS describing something that already happened
(a search, a cart change, or a checkout attempt that already succeeded or was already blocked).

Write a short, natural, friendly reply (2-4 sentences max) describing these facts to the user.

Hard rules:
- Never invent facts not present in the JSON. If something isn't in the JSON, don't mention it.
- If facts.type is "checkout" and facts.ok is false, clearly explain WHY it was blocked
  (use facts.reason / facts.detail) -- do not soften or hide that it failed.
- If facts.type is "checkout" and facts.ok is true, mention the amount and that no PIN/OTP
  was needed because it was within the pre-authorized mandate.
- For search_results, list item names with prices; if items is empty, say nothing matched.
- Keep it concise. No markdown headers, no emoji spam.
"""


def _checkout_failure_text(f: Dict) -> str:
    reason = f.get("reason")
    if reason == "empty_cart":
        return "Your cart is empty, nothing to check out."
    if reason == "no_mandate":
        return "You haven't authorized a spending mandate yet. Set one up first before I can check out on your behalf."
    return f"I stopped this purchase: {f.get('detail', reason)}"


def _fallback_compose(facts: Dict) -> str:
    t = facts.get("type")
    if t == "search_results":
        items = facts.get("items", [])
        if not items:
            return "I couldn't find anything matching that. Try a different search?"
        lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in items]
        return "Here's what I found:\n" + "\n".join(lines) + "\n\nSay 'add <sku id>' to add one to your cart."
    if t == "add_to_cart":
        if not facts.get("ok"):
            return "I couldn't find that item id. Try searching first."
        item = facts["added_item"]
        return f"Added {item['name']} (Rs {item['price']}) to your cart. Cart total: Rs {facts['cart_total']}."
    if t == "remove_from_cart":
        if not facts.get("ok"):
            return "Tell me the sku id to remove, e.g. 'remove sku_003'."
        return f"Removed. Cart total is now Rs {facts['cart_total']}."
    if t == "view_cart":
        cart = facts.get("cart", [])
        if not cart:
            return "Your cart is empty."
        lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in cart]
        return "Your cart:\n" + "\n".join(lines) + f"\n\nTotal: Rs {facts['cart_total']}"
    if t == "checkout":
        if facts.get("ok"):
            return (
                f"Done — debited Rs {facts['amount']} against your mandate with zero taps. "
                f"Transaction id {facts['txn_id']}. No PIN prompt needed, it was within your pre-authorized limit."
            )
        return _checkout_failure_text(facts)
    return (
        "Hi! I can search products, manage your cart, and check out for you "
        "once you've authorized a spending mandate. Try: 'show me earbuds under 3000'."
    )


def node_compose_reply(state: AgentState) -> AgentState:
    facts = state.get("facts", {"type": "chit_chat"})
    state["action_result"] = facts  # frontend still gets the raw structured facts too
    if GEMINI_API_KEY:
        try:
            reply = _call_gemini(COMPOSE_SYSTEM_PROMPT, json.dumps(facts), max_tokens=200)
            if reply:
                state["reply"] = reply
                return state
        except Exception as e:
            logger.warning("Gemini compose_reply failed, falling back to templates: %s", e)
            # fall through to deterministic fallback below
    state["reply"] = _fallback_compose(facts)
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


def run_agent(user_message: str, cart: List[Dict], mandate_id: Optional[str]) -> AgentState:
    initial: AgentState = {
        "user_message": user_message,
        "cart": cart,
        "mandate_id": mandate_id,
    }
    return graph.invoke(initial)
