# Project Rules for Claude Code

## PII & Security

This project stores and processes children's personal data (name, birthday, gender, reading
level). Treat all child/family data as high-sensitivity PII.

### Logging rules

- **Never log PII values in `logger.*` calls.** This includes: child names, birth dates,
  genders, reading interests, goals, user messages, family member names, and any field from
  `ChildProfile`, `FamilyMember`, `ChildReadingProfile`, `FamilyReadingPolicy`.
- When logging exceptions that may have touched DB rows or user input, log only the exception
  **type** (`type(exc).__name__`), never the full exception object or message.
  ```python
  # WRONG
  logger.warning("failed: %s", exc)
  # RIGHT
  logger.warning("failed: %s", type(exc).__name__)
  ```
- Safe to log: IDs (UUIDs), capability names, intent names, operation names, row counts,
  boolean flags.

### Prompt / LLM rules

- Do not add `logger.*` calls that print the raw `state["messages"]` or any user-supplied
  string without explicitly stripping PII first.
- When adding new LLM nodes, never pass raw DB rows into a prompt; serialize only the fields
  the node needs.

### Authorization rules

- Every repository read method that takes a `child_id` or `member_id` **must also filter by
  `family_id`**. A query scoped only to `child_id` is a cross-family data leak.
- Domain tools must always read identity from `current()` (the contextvar), never from
  user-supplied tool arguments.

## Testing rules

- New repository methods that read data must have a cross-family isolation test: seed data
  under family A, query with family B's id, assert empty result.
- Prompt-injection scenarios for any new LLM node that takes user input: verify that an
  off-roster or malformed LLM output is rejected by the post-LLM gating logic.
