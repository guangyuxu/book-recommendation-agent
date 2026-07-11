"""Memory subgraph: the agent's long-term-memory write path (decide -> confirm -> persist).

A self-contained feature package: the subgraph builder (graph.py), its nodes (decide, confirm,
profile_update), the pure confirm policy (confirm_policy), and its contracts (schemas) all live
here. The main graph (agent.graph) mounts `memory_graph` as one node that runs in parallel with
the answer pipeline.
"""

from .graph import memory_graph

__all__ = ["memory_graph"]
