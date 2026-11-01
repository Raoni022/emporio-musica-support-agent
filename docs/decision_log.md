# Decision Log — Empório da Música Support Agent

This log records how the design evolved before implementation. Planning was done in a conversation with Claude (chat): I reviewed the case, questioned each part of the design, and the specification changed in response. The resulting spec (`CLAUDE.md`) then guided implementation with Claude Code, one reviewed commit at a time.

Each entry: **what I questioned → what changed → why.**

## 2026-10-31 — Planning session

**1. Initial architecture**
- Started from: function calling over deterministic tools for the CSV data, the full policy manual in the system prompt, CLI interface, in-memory history.
- Rationale: the case explicitly asks for the simplest solution; the manual is only 8 pages.

**2. "Add more rules based on the brief and the manual."**
- Changed: added deterministic tools for return eligibility (regret / defect / preference windows), shipping options and store hours (with Brazilian and MS holidays). Added rules for privacy (LGPD), no checkout or stock reservation, seasonal campaigns only when active in the system, ambiguity handling, complaints and prompt-injection resistance.
- Why: these are recurring questions the manual covers, and dates and amounts must never be computed by the LLM.

**3. "Let's think harder about the technical decisions."**
- Changed: policies moved from "whole manual in the prompt" to a `get_policy(topic)` tool over the manual's numbered sections.
- Why: the brief evaluates whether the agent knows *when* to consult policies. As a tool, that decision becomes explicit, visible in traces and measurable. Splitting by the manual's own sections avoids retrieval errors without embeddings.
- Kept: plain function-calling loop instead of LangGraph (one agent, no persistent state yet; LangGraph is the next step when human handoff and conversation state appear).

**4. "What about libraries, observability and the long term?"**
- Changed: minimal dependency set, each one justified; added a local JSONL trace per turn (tools, latency, tokens, estimated cost) with e-mail and phone masked; added a "Path to production" section built on the idea that tools are a stable contract (CSV today, ERP API tomorrow, same agent).

**5. "Should it handle audio, documents or integrations? What about policy and data updates?"**
- Changed: audio, images and real integrations are explicitly out of scope (the brief asks for a *text* agent), each with a documented evolution path. The manual version is captured and returned by `get_policy`; startup fails if the manual cannot be split into its 10 sections.

**6. "Should we normalize, fix or improve the data?"**
- Changed: raw data is immutable; formats are normalized at load time; business facts are never "fixed" (suppress, keep as recorded, or choose and document); a data quality report is generated for the store.
- Running checks on the data at this point surfaced new issues: a duplicated customer e-mail, the `strings` spec meaning material for acoustic instruments but count for electric ones, specs contradicting descriptions, a "coming soon" product inside two orders, and duplicated promotions per product.

**7. "Do a deep audit against the job and the brief."**
- Changed: `OPENAI_BASE_URL` so any OpenAI-compatible provider (including free or local ones) runs without code changes; section 4.4 exclusions added to return eligibility; agent-loop tests with a fake LLM; pinned dependencies; explicit priorities (must / should / could) to avoid over-engineering.
- Also fixed: internal inconsistencies in the spec itself (section count, an eval example that contradicted the policy-as-tool design).

**8. "How do business rules get into the prompt?"**
- Changed: a rule-placement principle — calculable or verifiable rules live in code, informational rules come from `get_policy`, "when to use" lives in tool descriptions, and only behavioral rules live in the prompt. The prompt does not repeat rules the code already enforces.

**9. Persona**
- Decided: the assistant is called **Tom** (a play on a song's "tom", its key).

**10. "What if it doesn't know the answer, times out, the integration fails or there are too many requests?"**
- Changed: a failure-handling table — timeouts and retries with backoff, a deterministic fallback message with the store's contact info when the LLM is unavailable, structured tool errors, iteration and time budgets, a message-length cap and, for the optional API, a per-session rate limit.

**11. "Is the spec complete?"**
- Final review fixed: a parameter name inconsistent with the normalized specs, a missing `list_active_promotions` tool, a reference date without a time (needed for "are you open now?"), and an explicit per-step `pytest` run.

## Implementation (to be filled during the build)

Record here any point where I disagreed with generated code and changed it, and why.

- …
