"""Graph wiring.

    router -> investigate -> decide -> policy_check
    policy_check --approved--> verify
    policy_check --rejected/requires_human--> act_or_escalate
    verify --> act_or_escalate
    act_or_escalate -> respond -> END

Both of the documented `verify` edges land on `act_or_escalate`, so a plain edge
is equivalent to the two labelled ones - the node itself branches on `verified`
from state.

Note what is *not* here: there is no edge from `decide` to `act_or_escalate`, and
no edge from `investigate` or `router` either. `policy_check` and `verify` are the
only ways into the executor, which is the architectural rule that matters most -
no path reaches a money-moving action without passing the deterministic gate.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.llm import get_llm
from app.graph.state import CaseState

NODE_NAMES = (
    "router",
    "investigate",
    "decide",
    "policy_check",
    "verify",
    "act_or_escalate",
    "respond",
)

# The only nodes permitted to hand control to the executor.
GATE_NODES = ("policy_check", "verify")


def route_after_policy_check(state: CaseState) -> str:
    policy_result = state.get("policy_result") or {}
    return "verify" if policy_result.get("approved") else "act_or_escalate"


def build_graph(llm: Any = None, checkpointer: Any = None) -> Any:
    """Compile the graph.

    `llm` is injectable so tests can drive every node without an API key.
    `checkpointer` is threaded through for step 6 (PostgresSaver); step 3 runs
    without one.
    """
    llm = llm if llm is not None else get_llm()

    builder = StateGraph(CaseState)
    builder.add_node("router", partial(nodes.router_node, llm=llm))
    builder.add_node("investigate", partial(nodes.investigate_node, llm=llm))
    builder.add_node("decide", partial(nodes.decide_node, llm=llm))
    builder.add_node("policy_check", nodes.policy_check_node)
    builder.add_node("verify", partial(nodes.verify_node, llm=llm))
    builder.add_node("act_or_escalate", nodes.act_or_escalate_node)
    builder.add_node("respond", partial(nodes.respond_node, llm=llm))

    builder.add_edge(START, "router")
    builder.add_edge("router", "investigate")
    builder.add_edge("investigate", "decide")
    builder.add_edge("decide", "policy_check")
    builder.add_conditional_edges(
        "policy_check",
        route_after_policy_check,
        {"verify": "verify", "act_or_escalate": "act_or_escalate"},
    )
    builder.add_edge("verify", "act_or_escalate")
    builder.add_edge("act_or_escalate", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)
