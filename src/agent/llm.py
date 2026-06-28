"""Shared chat model: change the model (and its retry/timeout policy) in this one place only."""

from langchain.chat_models import init_chat_model

# max_retries: the Anthropic SDK retries 429 / 5xx / connection errors with exponential
#   backoff, so transient failures don't surface to the graph.
# timeout: per-request HTTP timeout in seconds; a hung request fails fast (and is then
#   retried) instead of blocking a node indefinitely.
model = init_chat_model(
    "claude-sonnet-4-6",
    temperature=0,
    max_retries=3,
    timeout=60,
)
