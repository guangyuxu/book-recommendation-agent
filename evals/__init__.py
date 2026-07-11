"""Eval harness for the book-recommendation agent.

Three strategies, one per subpackage:
- s1_classification: deterministic accuracy on the `understand` node's intent classification.
- s2_judge: LLM-as-judge scoring of generative capabilities (recommend / evaluate).
- s3_regression: end-to-end before/after comparison against committed baselines.

Design note (self-built engine, hybrid-shaped layout): datasets, scoring, and thresholds all
live in-repo so evals run offline and in CI with zero vendor lock-in. `_harness.report` keeps a
`report_to_langsmith` seam so results can later be pushed to a platform dashboard without moving
any files. See README.md.
"""
