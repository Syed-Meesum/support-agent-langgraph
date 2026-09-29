"""The mocked tools the graph is allowed to call.

Read tools (investigate node):
    get_customer, get_charges, get_subscription, search_knowledge_base

Action tools (act_or_escalate node, after policy_check + verify):
    issue_refund, cancel_subscription
"""

from app.tools.action_tools import cancel_subscription, issue_refund
from app.tools.customer_tools import get_charges, get_customer, get_subscription
from app.tools.errors import ToolError
from app.tools.knowledge_tools import search_knowledge_base

READ_TOOLS = (
    get_customer,
    get_charges,
    get_subscription,
    search_knowledge_base,
)
ACTION_TOOLS = (
    issue_refund,
    cancel_subscription,
)

__all__ = [
    "get_customer",
    "get_charges",
    "get_subscription",
    "search_knowledge_base",
    "issue_refund",
    "cancel_subscription",
    "ToolError",
    "READ_TOOLS",
    "ACTION_TOOLS",
]
