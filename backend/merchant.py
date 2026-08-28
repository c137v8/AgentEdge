"""
Dummy merchant profile.

The Agentic Commerce Protocol (ACP) checkout-session responses always
identify which merchant the agent is transacting with (business name,
support contact, accepted payment handlers, etc.) -- see
rfc.agentic_checkout.md and rfc.payment_handlers.md at
https://github.com/agentic-commerce-protocol/agentic-commerce-protocol.

This project has no real onboarded merchant, so these are realistic but
entirely fictional details for a fake storefront ("Nilgiri Basics"),
used purely to make ACP responses look like they came from a real,
ACP-registered business instead of an empty stub.
"""
from typing import Dict

MERCHANT: Dict = {
    "merchant_id": "acct_nilgiribasics_9f21ab",
    "business_name": "Nilgiri Basics Private Limited",
    "storefront_name": "Nilgiri Basics",
    # Merchant Category Code -- 5399 = "Miscellaneous General Merchandise",
    # a plausible MCC for a small general-goods storefront.
    "mcc": "5399",
    "url": "https://www.nilgiribasics.example.com",
    "support_email": "support@nilgiribasics.example.com",
    "support_phone": "+91-80-4567-1230",
    "business_address": {
        "name": "Nilgiri Basics Pvt Ltd",
        "line_one": "4th Floor, Prestige Tech Park",
        "line_two": "Kadubeesanahalli, Outer Ring Road",
        "city": "Bengaluru",
        "state": "KA",
        "country": "IN",
        "postal_code": "560103",
    },
    "gstin": "29AACCN1234F1Z5",  # fictional GST identification number
    # ACP spec version this merchant claims to implement -- date-based
    # versioning, see the spec's "API-Version" header convention.
    "acp_version": "2025-09-29",
    # This merchant hasn't onboarded with Stripe's Shared Payment Token
    # network -- it only accepts its own "seller-backed payment handler"
    # (see rfc.seller_backed_payment_handler.md), backed by the Reserve
    # Pay mandate engine in mandate.py. A real merchant would typically
    # also list "stripe" (Shared Payment Token) here.
    "accepted_payment_handlers": [
        {
            "type": "seller_backed",
            "id": "reserve_pay_mandate",
            "display_name": "UPI Reserve Pay (mandate)",
        }
    ],
}
