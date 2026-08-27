"""
LangGraph shopping agent.

Graph shape:

    interpret --> search        --> respond
              --> cart_update   --> respond
              --> checkout      --> respond
              --> chit_chat     --> respond

`interpret` uses Claude (Anthropic API) if ANTHROPIC_API_KEY is set, for
robust intent parsing + a natural reply. If no key is configured, it falls
back to a small rule-based parser so the demo still runs end-to-end offline
-- useful when showing this on a laptop with no internet during the
interview.

The important architectural point (say this out loud in the interview):
the LLM is ONLY ever allowed to *propose* an action (search / add_to_cart /
checkout). It never touches money directly. Every checkout proposal is
re-validated in code against the mandate guardrails in mandate.py before a
single rupee moves. That separation -- LLM proposes, deterministic code
disposes -- is the whole point of the "risk manager" framing.
"""
from __future__ import annotations
import json
import os
import re
import uuid
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

import inventory
import mandate as mandate_mod

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


class AgentState(TypedDict, total=False):
    user_message: str
    mandate_id: Optional[str]
    cart: List[Dict]
    intent: str
    slots: Dict
    reply: str
    action_result: Optional[Dict]


# --------------------------------------------------------------------------
# Node: interpret user message -> {intent, slots}
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the NLU layer of a shopping agent. Read the user's message and the
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

    # IMPORTANT: order matters here. "add sku_001 to cart" contains the word
    # "cart", so a naive check-for-"cart"-first would misclassify this as
    # "view_cart" instead of "add_to_cart". Checking for an explicit add/buy
    # verb (or a referenced sku id) BEFORE the generic cart-viewing check
    # avoids that. This was caught during testing -- see CHALLENGES.md.
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
    # default: treat as a search
    price_match = re.search(r"under\s*(?:rs\.?|₹)?\s*(\d+)", m)
    return {"intent": "search", "query": message, "max_price": int(price_match.group(1)) if price_match else None, "sku_id": ""}


def _claude_interpret(message: str, cart: List[Dict]) -> Dict:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Cart: {json.dumps(cart)}\nMessage: {message}"}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```json|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _rule_based_interpret(message)


def node_interpret(state: AgentState) -> AgentState:
    message = state["user_message"]
    cart = state.get("cart", [])
    if ANTHROPIC_API_KEY:
        try:
            parsed = _claude_interpret(message, cart)
        except Exception:
            parsed = _rule_based_interpret(message)
    else:
        parsed = _rule_based_interpret(message)
    state["intent"] = parsed.get("intent", "chit_chat")
    state["slots"] = parsed
    return state


# --------------------------------------------------------------------------
# Node: search
# --------------------------------------------------------------------------

def node_search(state: AgentState) -> AgentState:
    slots = state["slots"]
    results = inventory.search(slots.get("query", ""), max_price=slots.get("max_price"))
    state["action_result"] = {"type": "search_results", "items": results}
    if results:
        lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in results[:5]]
        state["reply"] = "Here's what I found:\n" + "\n".join(lines) + "\n\nSay 'add <sku id>' to add one to your cart."
    else:
        state["reply"] = "I couldn't find anything matching that. Try a different search?"
    return state


# --------------------------------------------------------------------------
# Node: cart operations
# --------------------------------------------------------------------------

def node_cart_update(state: AgentState) -> AgentState:
    slots = state["slots"]
    cart = state.get("cart", [])
    intent = state["intent"]

    if intent == "add_to_cart":
        sku_id = slots.get("sku_id") or ""
        match = re.search(r"sku_\d+", state["user_message"].lower())
        if not sku_id and match:
            sku_id = match.group(0)
        item = inventory.get_by_id(sku_id)
        if not item:
            state["reply"] = "I couldn't find that item id. Try searching first."
            return state
        cart.append(item)
        state["cart"] = cart
        state["reply"] = f"Added {item['name']} (Rs {item['price']}) to your cart. Cart total: Rs {sum(i['price'] for i in cart)}."

    elif intent == "remove_from_cart":
        match = re.search(r"sku_\d+", state["user_message"].lower())
        if match and cart:
            cart = [i for i in cart if i["id"] != match.group(0)]
            state["cart"] = cart
            state["reply"] = f"Removed. Cart total is now Rs {sum(i['price'] for i in cart)}."
        else:
            state["reply"] = "Tell me the sku id to remove, e.g. 'remove sku_003'."

    elif intent == "view_cart":
        if not cart:
            state["reply"] = "Your cart is empty."
        else:
            lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in cart]
            state["reply"] = "Your cart:\n" + "\n".join(lines) + f"\n\nTotal: Rs {sum(i['price'] for i in cart)}"

    state["action_result"] = {"type": "cart", "cart": cart}
    return state


# --------------------------------------------------------------------------
# Node: checkout (the guarded chokepoint)
# --------------------------------------------------------------------------

def node_checkout(state: AgentState) -> AgentState:
    cart = state.get("cart", [])
    mandate_id = state.get("mandate_id")
    if not cart:
        state["reply"] = "Your cart is empty, nothing to check out."
        return state
    if not mandate_id:
        state["reply"] = "You haven't authorized a spending mandate yet. Set one up first (top right) before I can check out on your behalf."
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
        state["action_result"] = {"type": "checkout_success", "txn": entry.id, "amount": total}
        state["reply"] = (
            f"Done — debited Rs {total} against your mandate with zero taps. "
            f"Transaction id {entry.id}. No PIN prompt needed, it was within your pre-authorized limit."
        )
        state["cart"] = []
    except mandate_mod.DebitError as e:
        state["action_result"] = {"type": "checkout_blocked", "code": e.code}
        state["reply"] = f"I stopped this purchase: {e.message}"
    return state


# --------------------------------------------------------------------------
# Node: chit chat
# --------------------------------------------------------------------------

def node_chit_chat(state: AgentState) -> AgentState:
    state["reply"] = (
        "Hi! I can search products, manage your cart, and check out for you "
        "once you've authorized a spending mandate. Try: 'show me earbuds under 3000'."
    )
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

    g.set_entry_point("interpret")
    g.add_conditional_edges("interpret", route, {
        "search": "search",
        "cart_update": "cart_update",
        "checkout": "checkout",
        "chit_chat": "chit_chat",
    })
    g.add_edge("search", END)
    g.add_edge("cart_update", END)
    g.add_edge("checkout", END)
    g.add_edge("chit_chat", END)
    return g.compile()


graph = build_graph()


def run_agent(user_message: str, cart: List[Dict], mandate_id: Optional[str]) -> AgentState:
    initial: AgentState = {
        "user_message": user_message,
        "cart": cart,
        "mandate_id": mandate_id,
    }
    return graph.invoke(initial)
