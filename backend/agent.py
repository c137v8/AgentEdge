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
import json
import logging
import os
import re
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


def _rule_based_interpret(message: str) -> Dict:
    m = message.lower()
    has_sku = re.search(r"\bsku_\d+\b", m)

    # Order matters: "add sku_001 to cart" contains the word "cart", so a
    # naive check-for-"cart"-first would misclassify this as "view_cart".
    # Checking for an explicit add verb (or a referenced sku id) first
    # avoids that. Caught during testing -- see CHALLENGES.md.
    if any(w in m for w in ["checkout", "buy now", "buy it", "place order", "pay now", "confirm order"]):
        return {"intent": "checkout", "query": "", "max_price": None, "sku_id": ""}
    if any(w in m for w in ["remove", "delete item", "take out"]):
        return {"intent": "remove_from_cart", "query": "", "max_price": None, "sku_id": ""}
    if any(w in m for w in ["add", "buy", "get me", "i want", "i'll take"]):
        price_match = re.search(r"under\s*(?:rs\.?|₹)?\s*(\d+)", m)
        return {
            "intent": "add_to_cart",
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
"""


def _checkout_result_text(f: Dict) -> str:
    reason = f.get("reason")
    if reason == "empty_cart":
        return "Your cart is empty, nothing to check out."
    if reason == "no_mandate":
        return "You haven't authorized a spending mandate yet. Set one up in Payments before I can check out on your behalf."
    return f"I stopped this purchase: {f.get('detail', reason)}"


def _fallback_compose(facts: Dict) -> str:
    t = facts.get("type")
    if t == "search_results":
        items = facts.get("items", [])
        if not items:
            return "I couldn't find anything matching that. Try a different search?"
        lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in items]
        return "Here's what I found:\n" + "\n".join(lines) + "\n\nSay 'add <name or sku id>' to add one to your cart."
    if t == "add_to_cart":
        if facts.get("ok"):
            item = facts["added_item"]
            return f"Added {item['name']} (Rs {item['price']}) to your cart. Cart total: Rs {facts['cart_total']}."
        reason = facts.get("reason")
        if reason == "ambiguous_reference":
            opts = facts.get("options", [])
            return "Which one did you mean — " + " or ".join(opts) + "?"
        return "I couldn't find that item. Try naming it more specifically or searching first."
    if t == "remove_from_cart":
        if not facts.get("ok"):
            return "Tell me the item to remove, e.g. 'remove sku_003'."
        return f"Removed. Cart total is now Rs {facts['cart_total']}."
    if t == "view_cart":
        cart = facts.get("cart", [])
        if not cart:
            return "Your cart is empty."
        lines = [f"- {i['name']} ({i['id']}) — Rs {i['price']}" for i in cart]
        return "Your cart:\n" + "\n".join(lines) + f"\n\nTotal: Rs {facts['cart_total']}"
    if t == "confirm_checkout":
        if facts.get("ok"):
            return f"Ready to check out {', '.join(facts['items'])} for Rs {facts['amount']}. Tap Confirm & Pay to complete it."
        return f"I can't check out yet: {facts.get('reason')}."
    if t == "checkout":
        if facts.get("ok"):
            merchant_bit = f" with {facts['merchant']}" if facts.get("merchant") else ""
            return (
                f"Done — charged Rs {facts['amount']}{merchant_bit} against your mandate with zero further taps. "
                f"Order {facts['txn_id']}. No PIN prompt needed, it was within your pre-authorized limit."
            )
        return _checkout_result_text(facts)
    return (
        "Hi! I can search products, manage your cart, and check out for you "
        "once you've authorized a spending mandate. Try: 'show me earbuds under 3000'."
    )


def compose_reply_from_facts(facts: Dict) -> str:
    if GEMINI_API_KEY:
        try:
            reply = _call_gemini(COMPOSE_SYSTEM_PROMPT, json.dumps(facts), max_tokens=200)
            if reply:
                return reply
        except Exception as e:
            logger.warning("Gemini compose_reply failed, falling back to templates: %s", e)
    return _fallback_compose(facts)


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
