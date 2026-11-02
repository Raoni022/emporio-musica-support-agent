# Decision Log — Empório da Música Support Agent

This log records how the design evolved before and during implementation. Planning was done in a conversation with Claude (chat): I reviewed the case, questioned each part of the design, and the specification changed in response. The resulting spec (`CLAUDE.md`) then guided implementation with Claude Code, one reviewed commit at a time.

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

## 2026-10-31 to 2026-11-02 — Implementation

Claude Code implemented the spec one step per commit; I reviewed each step, ran it and made every commit myself. Entries below are the points where I questioned what was generated, or where testing it myself changed the plan.

**12. "Did you put these up for me to validate?"**
- Context: in step 4 the summary listed three decisions the spec did not cover (search filters by final price, alternatives within ±30%, PIX left for step 5) without making clear they were Claude Code's own choices.
- Changed: a standing rule for every step: decisions not in the spec are flagged in the step summary, I validate them by committing or object before committing, and each one goes into the README's assumptions table in the same commit.
- Why: the brief asks for documented assumptions, and I want to be able to defend every choice in the repository as mine.

**13. "Are the commit messages good?"**
- Found: commits 2 to 5 had 6–8 body lines, above the spec's 2–5.
- Decided: keep the rule for the remaining commits and leave the published history as it is; rewriting it would contradict the "never rewrite history" rule, which matters more.

**14. Manual contradictions: keep `notes` in `get_policy`.**
- The manual contradicts itself (two WhatsApp numbers, an accented e-mail, two installment minimums). I chose to keep the manual text untouched and return a `notes` field saying which value to use, instead of putting manual content into the prompt.
- Why: it keeps the spec's rule that the manual never goes into the prompt, and the correction travels with the section it corrects.

**15. Checking the recruiter's e-mail against the brief.**
- Found: the brief says to reply with the repository link; the e-mail asks for a PDF through Google Forms and says all responses must be in English; the deadline is Tuesday 2026-11-03.
- Changed: every example conversation opens with an English summary (the conversation stays in Portuguese, the customers' language).

**16. First real run with my API key.**
- Found three bugs the fake-LLM tests could not catch: an empty `OPENAI_BASE_URL` in `.env` broke every call; an old `OPENAI_API_KEY` in my Windows environment silently won over `.env`; and the model computed 10 installments with the previous product's price, because it called the installments tool in parallel with the product lookup.
- Changed: three `fix:` commits; `.env` now wins over system variables, and `calculate_installments` looks up the product's price itself.
- Why: the installment bug quoted a value that a tool really returned, so not even the price guardrail would have caught it. It had to be fixed in the tool.

**17. "He got the day wrong, and there are other errors."** (my own CLI session)
- The day was the fixed reference date (2026-03-25). Decided: keep it fixed by default, since with the real date every delivered order is outside any return window and the brief's return question would always get "no".
- The other errors were real: "violão clássico" returned nothing and the agent told me there were no classical guitars (there are 15); repeated status lines; eight product lookups to answer "any on sale?"; offering to "close the order", which the agent cannot do.
- Changed: a search fix (Portuguese terms derived from the specs, and free-text words no longer cancel the structured filters), the status-line fix, and the rest went to the eval before touching the prompt.

**18. Eval-driven fixes.**
- The first eval run scored 17/20; prompt-only fixes went to 16/20 with different failures. Instructions placed inside the tool results (next to the data) made the behavior stable: 20/20 in the last run, 79/80 over four runs.
- Not fixed, documented as limitations: the redundant per-product lookups, and the 4.4 return exclusions not always being raised.
