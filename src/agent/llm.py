"""Shared chat model: change the model (and its retry/timeout policy) in this one place only."""

from langchain.chat_models import init_chat_model

# max_retries: the Anthropic SDK retries 429 / 5xx / connection errors with exponential
#   backoff, so transient failures don't surface to the graph.
# timeout: per-request HTTP timeout in seconds; a hung request fails fast (and is then
#   retried) instead of blocking a node indefinitely.
# disable_streaming: LangGraph attaches streaming callbacks to every node, so the model would
#   otherwise stream token-by-token and go through generate_from_stream. That path raises
#   "No generations found in stream." when a stream yields no chunks -- seen on the
#   with_structured_output (forced tool-calling) calls. We consume only whole responses
#   (model.invoke -> full AIMessage / structured object; no token streaming to the client),
#   so disabling streaming removes that failure mode with no UX cost.
model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0,
    max_retries=3,
    timeout=60,
    disable_streaming=True,
)
